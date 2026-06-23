"""EEG Artifact Removal Module."""

import mne
import numpy as np
from typing import Dict, List, Tuple

from mne.preprocessing import ICA


def detect_bad_channels(raw: mne.io.Raw) -> Tuple[mne.io.Raw, Dict[str, List[str]]]:
    """Detect and mark bad EEG channels using flatline, correlation, and line-noise criteria.

    Args:
        raw: Raw EEG data.

    Returns:
        Tuple of (raw with bad channels marked in ``raw.info['bads']``,
        dict mapping criterion name to the list of channels it flagged).
    """
    if raw.info["bads"] is None:
        raw.info["bads"] = []

    # Flatline: channels with near-zero variance
    flatline_channels: List[str] = []
    flatline_threshold: float = 1e-6

    for ch in raw.ch_names:
        if ch not in raw.info["bads"]:
            ch_data = raw.get_data(picks=[ch])[0]
            if np.std(ch_data) < flatline_threshold:
                flatline_channels.append(ch)

    # Correlation: channels whose signal is dominated by a single neighbour
    correlation_channels: List[str] = []
    correlation_threshold: float = 0.8

    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False)
    eeg_channels: List[str] = [raw.ch_names[i] for i in eeg_picks]

    if len(eeg_channels) > 1:
        eeg_data = raw.get_data(picks=eeg_picks)
        corr_matrix = np.corrcoef(eeg_data)
        for i in range(len(eeg_channels)):
            for j in range(i + 1, len(eeg_channels)):
                if abs(corr_matrix[i, j]) > correlation_threshold:
                    candidate: str = eeg_channels[j]
                    if candidate not in raw.info["bads"] + flatline_channels:
                        correlation_channels.append(candidate)

    # Line noise: channels with anomalously high 50 Hz power
    line_noise_channels: List[str] = []
    line_noise_threshold: float = 5.0

    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False)
    if len(eeg_picks) > 0:
        psd, freqs = mne.time_frequency.psd_welch(
            raw, fmin=45, fmax=55, tmax=60, picks=eeg_picks, verbose=False
        )
        freq_50_idx: int = int(np.argmin(np.abs(freqs - 50)))
        power_50hz = psd[:, freq_50_idx]
        z_scores = (power_50hz - np.mean(power_50hz)) / np.std(power_50hz)
        for i, ch_idx in enumerate(eeg_picks):
            if z_scores[i] > line_noise_threshold:
                ch_name: str = raw.ch_names[ch_idx]
                already_bad: List[str] = raw.info["bads"] + flatline_channels + correlation_channels
                if ch_name not in already_bad:
                    line_noise_channels.append(ch_name)

    all_bad: List[str] = list(set(flatline_channels + correlation_channels + line_noise_channels))
    raw.info["bads"].extend(all_bad)

    rejected_info: Dict[str, List[str]] = {
        "flatline": flatline_channels,
        "correlation": correlation_channels,
        "line_noise": line_noise_channels,
        "all": all_bad,
    }
    return raw, rejected_info


def apply_asr(raw: mne.io.Raw, cutoff: float = 20.0) -> mne.io.Raw:
    """Apply Artifact Subspace Reconstruction to continuous EEG data.

    Calibrates ASR on segments not covered by BAD annotations, then
    reconstructs burst artefacts in place across the full recording.
    The pre/post/between-block BAD annotations added by
    ``reject_experimental_blocks`` therefore naturally confine calibration
    to within-experiment data.

    Args:
        raw: Filtered and resampled raw EEG data with BAD annotations
             already set for non-experimental periods.
        cutoff: ASR cutoff in standard deviations. Lower values are more
                aggressive. 20 is the recommended starting point.

    Returns:
        Raw data with burst artefacts reconstructed in place.
    """
    from asrpy import ASR

    asr: ASR = ASR(sfreq=raw.info["sfreq"], cutoff=cutoff)
    asr.fit(raw)
    raw = asr.transform(raw)
    return raw


def apply_line_noise_removal(raw: mne.io.Raw) -> mne.io.Raw:
    """Remove 50 Hz line noise and harmonics via spectrum_fit notch filtering.

    Args:
        raw: Raw EEG data.

    Returns:
        Raw data with 50, 100, and 150 Hz components removed.
    """
    raw.notch_filter(freqs=[50, 100, 150], method="spectrum_fit", notch_widths=2, verbose=False)
    return raw


def ica_pipeline(raw: mne.io.Raw, n_components: int = 20) -> ICA:
    """Run ICA decomposition and automatically flag EOG and ECG artefact components.

    Args:
        raw: Raw EEG data.
        n_components: Number of ICA components to compute.

    Returns:
        Fitted ICA object with artefact components flagged in ``ica.exclude``.
    """
    ica: ICA = ICA(
        n_components=n_components,
        random_state=97,
        method="picard",
        max_iter="auto",
        verbose=False,
    )
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False)
    ica.fit(raw, picks=eeg_picks)

    try:
        if len(mne.pick_types(raw.info, eog=True)) > 0:
            ica.find_bads_eog(raw, verbose=False)
        if len(mne.pick_types(raw.info, ecg=True)) > 0:
            ica.find_bads_ecg(raw, verbose=False)
    except Exception as e:
        print(f"Automatic artefact detection failed: {e}")

    return ica


def apply_ica_rejection(raw: mne.io.Raw, ica: ICA, bad_components: List[int]) -> mne.io.Raw:
    """Remove flagged ICA components from continuous data.

    Args:
        raw: Raw EEG data.
        ica: Fitted ICA object.
        bad_components: Component indices to exclude.

    Returns:
        Raw data with artefact components projected out.
    """
    if bad_components:
        ica.exclude = bad_components
        raw = ica.apply(raw)
    return raw
