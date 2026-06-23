#!/usr/bin/env python3
"""Two-pass EEG Preprocessing Pipeline — Prior RDM Study.

Pass 1 (--stage ica)
    Preprocesses with a 1 Hz high-pass filter, fits ICA, and saves the
    components for human review. Produces:
      {subject}_ica.fif              — fitted ICA object
      {subject}_ica_components.png   — topomap of all components
      {subject}_ica_review.json      — auto-detected exclusions + review flag

    After running Pass 1:
      1. Inspect the component topomap and the ICA .fif in your viewer.
      2. Open {subject}_ica_review.json.
      3. Edit confirmed_exclude if the auto-detection missed or over-flagged.
      4. Set "reviewed": true.
      5. Run Pass 2.

Pass 2 (--stage final)
    Reloads the raw data, preprocesses with the ERP-appropriate 0.1 Hz
    high-pass, applies the reviewed ICA components, and extracts stimulus-
    and response-locked epochs.

Usage:
    python run_pipeline.py --subject ANA60 --stage ica
    # … review JSON …
    python run_pipeline.py --subject ANA60 --stage final

Pipeline steps (both passes):
    1.  Load BrainVision (filename-patch workaround)
    2.  Configure channels (HEOG/VEOG → EOG; add FCz)
    3.  Set standard_1020 montage
    4.  Extract events from annotations
    5.  High-pass + notch filter (before resample; hp_freq differs per pass)
    6.  Resample to 250 Hz
    7.  Re-extract events at new sfreq
    8.  Mark pre-/post-experiment segments as BAD
    9.  Bad-channel detection (flatline)
    10. Average reference

Pass 1 continues:
    11. Fit ICA (20 components, Picard, seed=97; auto EOG detection)
    12. Save ICA + component plots + review JSON  →  human review

Pass 2 continues:
    11. Restore bad channels from review JSON
    12. Apply confirmed ICA components + interpolate bad channels
    13. Extract stimulus-locked epochs  (−200 to 1000 ms)
    14. Extract response-locked epochs (−1000 to 500 ms)
    15. Save checkpoints + CPP QC figure
"""

import argparse
import json
import re
import shutil
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parents[3]  # repo root
EEG_DIR = Path("/mnt/e/priorRDM/Study/EEGData")  # raw data on external drive (WSL path)
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "eeg"

SFREQ_TARGET: int = 250
HP_FREQ: float = 0.1       # Hz — final preprocessing pass (ERP-appropriate)
HP_FREQ_ICA: float = 1.0   # Hz — ICA identification pass (removes slow drifts)
NOTCH_FREQS: list[int] = [50, 100, 150]
N_ICA: int = 20
ICA_SEED: int = 97
TMIN, TMAX = -0.200, 1.000
BASELINE = (-0.200, 0.0)
REJECT_CRIT: dict[str, float] = {"eeg": 100e-6, "eog": 200e-6}

ONSET_CODES: list[int] = (
    list(range(31, 35))
    + list(range(41, 45))
    + list(range(51, 55))
    + list(range(61, 65))
)

BLOCK_START_CODES: list[int] = list(range(1, 9))
BLOCK_END_CODES: list[int] = list(range(11, 19))

COND_MAP: dict[int, str] = {3: "Mono", 4: "Di_null", 5: "Di_part", 6: "Di_full"}
COH_MAP: dict[int, float] = {1: 0.0, 2: round(2 / 30, 4), 3: round(4 / 30, 4), 4: round(10 / 30, 4)}
COH_LABEL: dict[int, str] = {1: "0%", 2: "6.7%", 3: "13.3%", 4: "33.3%"}

RESP_CODES: list[int] = (
    list(range(131, 139))
    + list(range(141, 149))
    + list(range(151, 159))
    + list(range(161, 169))
)
RESP_TMIN, RESP_TMAX = -1.000, 0.500
RESP_BASELINE = (-1.000, -0.800)

CPP_CHANNELS: list[str] = ["Pz", "CPz", "P1", "P2", "P3", "P4", "CP1", "CP2", "CP3", "CP4", "POz"]
CPP_STIM_WINDOW = (0.200, 1.000)
CPP_RESP_WINDOW = (-0.400, 0.050)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 · Load BrainVision (with filename-patch workaround)
# ─────────────────────────────────────────────────────────────────────────────


def load_brainvision(subject_id: str) -> mne.io.Raw:
    """Load a BrainVision recording, patching any internal filename mismatches.

    Some recordings have DataFile/MarkerFile headers pointing to a generic
    name while the actual files use the subject ID. The patch happens in a
    temp directory so the originals are never modified.

    Args:
        subject_id: Subject folder name under EEG_DIR.

    Returns:
        In-memory RawArray with annotations transferred from the source file.
    """
    subj_dir = EEG_DIR / subject_id
    vhdr_paths = list(subj_dir.glob("*.vhdr"))
    if not vhdr_paths:
        raise FileNotFoundError(f"No .vhdr found in {subj_dir}")

    vhdr_path = vhdr_paths[0]
    stem = vhdr_path.stem
    vmrk_path = vhdr_path.with_suffix(".vmrk")
    eeg_path = vhdr_path.with_suffix(".eeg")

    vhdr_text = vhdr_path.read_text(encoding="utf-8", errors="replace")
    vmrk_text = vmrk_path.read_text(encoding="utf-8", errors="replace")

    vhdr_fixed = re.sub(r"(?i)DataFile=.*", f"DataFile={stem}.eeg", vhdr_text)
    vhdr_fixed = re.sub(r"(?i)MarkerFile=.*", f"MarkerFile={stem}.vmrk", vhdr_fixed)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / f"{stem}.vhdr").write_text(vhdr_fixed, encoding="utf-8")
        (tmp / f"{stem}.vmrk").write_text(vmrk_text, encoding="utf-8")
        shutil.copy(eeg_path, tmp / f"{stem}.eeg")

        _raw = mne.io.read_raw_brainvision(str(tmp / f"{stem}.vhdr"), preload=True, verbose=False)
        raw = mne.io.RawArray(_raw.get_data(), _raw.info, verbose=False)
        raw.set_annotations(_raw.annotations)

    print(f"  Loaded: {len(raw.ch_names)} ch, {raw.info['sfreq']:.0f} Hz, {raw.times[-1]/60:.1f} min")
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 · Channel configuration
# ─────────────────────────────────────────────────────────────────────────────


def configure_channels(raw: mne.io.Raw) -> mne.io.Raw:
    """Set EOG channel types and add FCz as a flat reference channel."""
    eog_chs: list[str] = [c for c in ("HEOG", "VEOG") if c in raw.ch_names]
    if eog_chs:
        raw.set_channel_types({c: "eog" for c in eog_chs})
        print(f"  EOG channels: {eog_chs}")

    if "FCz" not in raw.ch_names:
        fcz_info = mne.create_info(["FCz"], raw.info["sfreq"], ch_types="eeg")
        fcz_raw = mne.io.RawArray(np.zeros((1, len(raw.times))), fcz_info, verbose=False)
        raw.add_channels([fcz_raw], force_update_info=True)
        print("  Added FCz (zero reference channel)")

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 · Montage
# ─────────────────────────────────────────────────────────────────────────────


def set_montage(raw: mne.io.Raw) -> mne.io.Raw:
    """Apply standard_1020 montage, silently ignoring unrecognised channels."""
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="ignore", verbose=False)
    print("  standard_1020 montage set")
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 · Events from annotations
# ─────────────────────────────────────────────────────────────────────────────


def get_events(raw: mne.io.Raw) -> tuple[np.ndarray, dict[str, int]]:
    """Extract events from BrainVision annotations.

    Returns:
        Tuple of (events array, event_id dict).
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    unique: list[int] = sorted(set(events[:, 2].tolist()))
    print(f"  {len(events)} events, {len(unique)} unique codes: {unique}")
    return events, event_id


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 · Filtering  (applied before resample so notch at 150 Hz is valid)
# ─────────────────────────────────────────────────────────────────────────────


def apply_filters(raw: mne.io.Raw, hp_freq: float = HP_FREQ) -> mne.io.Raw:
    """Apply high-pass and notch filters to continuous data.

    Must be called before resampling to 250 Hz so the 150 Hz notch is within
    the Nyquist limit of the original (typically ~1000 Hz) recording.

    Args:
        raw: Raw EEG data at original sampling rate.
        hp_freq: High-pass cut-off in Hz. Use HP_FREQ_ICA (1.0) for the ICA
                 identification pass and HP_FREQ (0.1) for the final pass.

    Returns:
        Filtered raw data.
    """
    raw.filter(
        l_freq=hp_freq,
        h_freq=None,
        method="fir",
        fir_design="firwin",
        phase="zero",
        verbose=False,
    )
    print(f"  HP filter: {hp_freq} Hz (FIR, zero-phase)")

    raw.notch_filter(NOTCH_FREQS, method="fir", verbose=False)
    print(f"  Notch filter: {NOTCH_FREQS} Hz")

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 · Resample
# ─────────────────────────────────────────────────────────────────────────────


def resample(raw: mne.io.Raw) -> mne.io.Raw:
    """Downsample to SFREQ_TARGET Hz."""
    raw.resample(SFREQ_TARGET, npad="auto", verbose=False)
    print(f"  Resampled → {SFREQ_TARGET} Hz")
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 · Mark pre-/post-experiment segments as BAD
# ─────────────────────────────────────────────────────────────────────────────


def mark_experiment_bounds(raw: mne.io.Raw, events: np.ndarray) -> mne.io.Raw:
    """Annotate data outside the experimental period with BAD_ labels.

    Uses block-start triggers (1–8) or practice-start (200) to find the
    beginning, and the last response trigger to find the end.

    Args:
        raw: Resampled raw EEG data.
        events: Events array at current sfreq.

    Returns:
        Raw data with BAD_pre_experiment / BAD_post_experiment annotations.
    """
    sfreq: float = raw.info["sfreq"]
    rec_end: float = raw.times[-1]
    new_anns: list[tuple[float, float, str]] = []

    boundary_codes: list[int] = BLOCK_START_CODES + [200]
    starts = events[np.isin(events[:, 2], boundary_codes)]
    if len(starts):
        first_t: float = starts[0, 0] / sfreq
        pre_buffer: float = 2.0
        if first_t > pre_buffer:
            dur: float = first_t - pre_buffer
            new_anns.append((0.0, dur, "BAD_pre_experiment"))
            print(f"  BAD_pre_experiment: {dur:.1f} s")

    resp_codes: list[int] = list(range(130, 170)) + BLOCK_END_CODES + [201]
    ends = events[np.isin(events[:, 2], resp_codes)]
    if len(ends):
        last_t: float = ends[-1, 0] / sfreq
        post_buffer: float = 2.0
        if rec_end > last_t + post_buffer:
            onset: float = last_t + post_buffer
            post_dur: float = rec_end - onset
            new_anns.append((onset, post_dur, "BAD_post_experiment"))
            print(f"  BAD_post_experiment: {post_dur:.1f} s")

    if new_anns:
        extra = mne.Annotations(
            [a[0] for a in new_anns],
            [a[1] for a in new_anns],
            [a[2] for a in new_anns],
            orig_time=raw.annotations.orig_time,
        )
        raw.set_annotations(raw.annotations + extra)

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Step 9 · Bad channel detection
# ─────────────────────────────────────────────────────────────────────────────


def detect_bad_channels(raw: mne.io.Raw) -> mne.io.Raw:
    """Mark flatline EEG channels (SD < 0.5 µV) as bad in-place.

    FCz is excluded from the check since it is flat by design (online ref).

    Args:
        raw: Referenced raw EEG data.

    Returns:
        Raw data with bad channels added to raw.info['bads'].
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, exclude=[])
    eeg_chs: list[str] = [raw.ch_names[i] for i in eeg_picks]
    chs_check: list[str] = [c for c in eeg_chs if c != "FCz"]
    picks: list[int] = [raw.ch_names.index(c) for c in chs_check]

    if not picks:
        return raw

    data = raw.get_data(picks=picks)
    std = np.std(data, axis=1)
    flatline: list[str] = [chs_check[i] for i, s in enumerate(std) if s < 0.5e-6]

    if flatline:
        raw.info["bads"] = sorted(set(raw.info["bads"] + flatline))
        print(f"  Bad channels (flatline): {flatline}")
    else:
        print("  No flatline channels detected")

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Step 10 · Average reference
# ─────────────────────────────────────────────────────────────────────────────


def set_reference(raw: mne.io.Raw) -> mne.io.Raw:
    """Apply average reference across all EEG channels including FCz."""
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
    print("  Average reference applied (FCz included)")
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Shared preprocessing helper
# ─────────────────────────────────────────────────────────────────────────────


def _run_preprocessing(
    subject_id: str,
    hp_freq: float,
) -> tuple[mne.io.Raw, np.ndarray, dict[str, int]]:
    """Run steps 1–10 common to both pipeline passes.

    Args:
        subject_id: Subject folder name under EEG_DIR.
        hp_freq: High-pass cut-off in Hz. HP_FREQ_ICA for Pass 1, HP_FREQ
                 for Pass 2.

    Returns:
        Tuple of (preprocessed raw, events at 250 Hz, event_id dict).
    """
    raw: mne.io.Raw = load_brainvision(subject_id)
    raw = configure_channels(raw)
    raw = set_montage(raw)
    _events_orig, _event_id = get_events(raw)  # needed before filter/resample

    # Filter before resample so the 150 Hz notch remains within Nyquist
    raw = apply_filters(raw, hp_freq=hp_freq)
    raw = resample(raw)

    # Re-extract events at the new sfreq (sample numbers change after resample)
    events: np.ndarray
    event_id: dict[str, int]
    events, event_id = get_events(raw)

    raw = mark_experiment_bounds(raw, events)
    raw = detect_bad_channels(raw)
    raw = set_reference(raw)
    return raw, events, event_id


# ─────────────────────────────────────────────────────────────────────────────
# Pass 1 · ICA fitting
# ─────────────────────────────────────────────────────────────────────────────


def run_ica(raw: mne.io.Raw) -> mne.preprocessing.ICA:
    """Fit ICA and auto-detect EOG artefact components.

    Args:
        raw: Preprocessed raw data (ideally 1 Hz HP filtered).

    Returns:
        Fitted ICA with auto-detected components in ica.exclude.
    """
    ica: mne.preprocessing.ICA = mne.preprocessing.ICA(
        n_components=N_ICA,
        method="picard",
        random_state=ICA_SEED,
        max_iter=1000,
        verbose=False,
    )
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, exclude="bads")
    ica.fit(raw, picks=eeg_picks, verbose=False)
    print(f"  ICA fitted: {N_ICA} components (Picard, seed={ICA_SEED})")

    try:
        eog_idx: list[int]
        eog_idx, _ = ica.find_bads_eog(raw, verbose=False)
        ica.exclude = eog_idx
        print(f"  Auto-detected EOG components: {eog_idx}")
    except Exception as exc:
        print(f"  EOG auto-detection skipped: {exc}")

    return ica


def save_ica_review(
    subject_id: str,
    ica: mne.preprocessing.ICA,
    bad_channels: list[str],
) -> Path:
    """Save ICA, component topomaps, and a review JSON for human inspection.

    The review JSON contains the auto-detected component list under
    confirmed_exclude. The user edits this list if needed, then sets
    "reviewed": true before running Pass 2.

    Args:
        subject_id: Subject identifier.
        ica: Fitted ICA object with auto-detected components in ica.exclude.
        bad_channels: Flatline channels from Pass 1, carried to Pass 2 so
                      interpolation uses the same channel set.

    Returns:
        Path to the written review JSON.
    """
    out: Path = OUTPUT_DIR / subject_id
    out.mkdir(parents=True, exist_ok=True)

    ica_path: Path = out / f"{subject_id}_ica.fif"
    ica.save(str(ica_path), overwrite=True, verbose=False)
    print(f"  ICA saved: {ica_path.name}")

    # Component topomaps — one PNG per figure (MNE splits into pages)
    comp_path: Path = out / f"{subject_id}_ica_components.png"
    figs = ica.plot_components(show=False)
    if not isinstance(figs, list):
        figs = [figs]
    figs[0].savefig(str(comp_path), dpi=150, bbox_inches="tight")
    for f in figs:
        plt.close(f)
    print(f"  Component topomaps: {comp_path.name}")

    review: dict[str, Any] = {
        "subject_id": subject_id,
        "hp_freq_hz": HP_FREQ_ICA,
        "bad_channels": bad_channels,
        "ica_file": ica_path.name,
        "auto_exclude": list(ica.exclude),
        "confirmed_exclude": list(ica.exclude),
        "reviewed": False,
        "notes": "",
    }
    review_path: Path = out / f"{subject_id}_ica_review.json"
    review_path.write_text(json.dumps(review, indent=2))
    print(f"  Review JSON: {review_path.name}")

    return review_path


# ─────────────────────────────────────────────────────────────────────────────
# Pass 2 · Load reviewed ICA
# ─────────────────────────────────────────────────────────────────────────────


def load_ica_review(subject_id: str) -> dict[str, Any]:
    """Load and validate the human-reviewed ICA JSON.

    Args:
        subject_id: Subject identifier.

    Returns:
        Review dict containing confirmed_exclude, bad_channels, and ica_file.

    Raises:
        FileNotFoundError: If the review JSON does not exist (Pass 1 not run).
        RuntimeError: If "reviewed" is false (human review not completed).
    """
    review_path: Path = OUTPUT_DIR / subject_id / f"{subject_id}_ica_review.json"
    if not review_path.exists():
        raise FileNotFoundError(
            f"Review file not found: {review_path}\nRun --stage ica first."
        )

    review: dict[str, Any] = json.loads(review_path.read_text())

    if not review.get("reviewed", False):
        out: Path = OUTPUT_DIR / subject_id
        raise RuntimeError(
            f"ICA review not completed for {subject_id}.\n"
            f"  1. Inspect component topomaps: {out / (subject_id + '_ica_components.png')}\n"
            f"  2. Load ICA in MNE: mne.preprocessing.read_ica('{out / review['ica_file']}')\n"
            f"  3. Edit confirmed_exclude in: {review_path}\n"
            f"  4. Set \"reviewed\": true and re-run --stage final."
        )

    return review


# ─────────────────────────────────────────────────────────────────────────────
# Pass 2 · Apply ICA + interpolate
# ─────────────────────────────────────────────────────────────────────────────


def apply_ica_and_interpolate(raw: mne.io.Raw, ica: mne.preprocessing.ICA) -> mne.io.Raw:
    """Apply confirmed ICA exclusions and interpolate bad channels.

    Args:
        raw: Final-pass preprocessed raw data.
        ica: ICA object with confirmed_exclude set in ica.exclude.

    Returns:
        Cleaned raw data with bad channels interpolated.
    """
    if ica.exclude:
        ica.apply(raw, verbose=False)
        print(f"  ICA applied: removed components {ica.exclude}")
    else:
        print("  ICA: no components to remove")

    if raw.info["bads"]:
        bads_before: list[str] = raw.info["bads"][:]
        raw.interpolate_bads(reset_bads=True, verbose=False)
        print(f"  Interpolated bad channels: {bads_before}")

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Epoch extraction
# ─────────────────────────────────────────────────────────────────────────────


def build_trial_metadata(onset_events: np.ndarray, sfreq: float) -> pd.DataFrame:
    """Decode stimulus trigger codes into a per-trial metadata table.

    Args:
        onset_events: Events array filtered to stimulus onset codes.
        sfreq: Sampling frequency in Hz.

    Returns:
        DataFrame with condition, coherence, and timing columns.
    """
    rows: list[dict[str, Any]] = []
    for ev in onset_events:
        code: int = int(ev[2])
        cd: int = code // 10
        chd: int = code % 10
        rows.append(
            {
                "onset_code": code,
                "condition": COND_MAP.get(cd, "Unknown"),
                "coherence": COH_MAP.get(chd, np.nan),
                "coherence_label": COH_LABEL.get(chd, "Unknown"),
                "onset_sample": int(ev[0]),
                "onset_time_s": round(ev[0] / sfreq, 4),
            }
        )
    return pd.DataFrame(rows)


def extract_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: dict[str, int],
) -> mne.Epochs:
    """Extract stimulus-locked epochs (TMIN to TMAX) with amplitude rejection.

    Args:
        raw: Cleaned raw data.
        events: Events array at current sfreq.
        event_id: MNE event_id dict.

    Returns:
        Preloaded Epochs object with per-trial metadata.
    """
    onset_events: np.ndarray = events[np.isin(events[:, 2], ONSET_CODES)]
    onset_event_id: dict[str, int] = {k: v for k, v in event_id.items() if v in ONSET_CODES}
    metadata: pd.DataFrame = build_trial_metadata(onset_events, raw.info["sfreq"])

    epochs: mne.Epochs = mne.Epochs(
        raw,
        onset_events,
        event_id=onset_event_id,
        tmin=TMIN,
        tmax=TMAX,
        baseline=BASELINE,
        metadata=metadata,
        reject=REJECT_CRIT,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )

    n_kept: int = len(epochs)
    n_total: int = len(onset_events)
    pct: float = 100 * n_kept / n_total if n_total else 0
    print(f"  Epochs: {n_kept} / {n_total} kept ({pct:.1f} %)")
    for cond in ("Mono", "Di_null", "Di_part", "Di_full"):
        n = (epochs.metadata["condition"] == cond).sum() if epochs.metadata is not None else "?"
        print(f"    {cond}: {n}")

    return epochs


def build_response_metadata(resp_events: np.ndarray, sfreq: float) -> pd.DataFrame:
    """Decode response trigger codes into per-trial metadata.

    Args:
        resp_events: Events array filtered to response codes.
        sfreq: Sampling frequency in Hz.

    Returns:
        DataFrame with condition, coherence, accuracy, and timing columns.
    """
    RESP_COND_MAP: dict[int, str] = {3: "Mono", 4: "Di_null", 5: "Di_part", 6: "Di_full"}
    rows: list[dict[str, Any]] = []
    for ev in resp_events:
        code: int = int(ev[2])
        cd: int = (code - 100) // 10
        units: int = code % 10
        if 1 <= units <= 4:
            accuracy, coh_num = "correct", units
        elif 5 <= units <= 8:
            accuracy, coh_num = "incorrect", units - 4
        else:
            accuracy, coh_num = "other", 0
        rows.append(
            {
                "resp_code": code,
                "condition": RESP_COND_MAP.get(cd, "Unknown"),
                "coherence": COH_MAP.get(coh_num, np.nan),
                "coherence_label": COH_LABEL.get(coh_num, "Unknown"),
                "accuracy": accuracy,
                "resp_sample": int(ev[0]),
                "resp_time_s": round(ev[0] / sfreq, 4),
            }
        )
    return pd.DataFrame(rows)


def extract_response_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: dict[str, int],
) -> Optional[mne.Epochs]:
    """Extract response-locked epochs (RESP_TMIN to RESP_TMAX).

    Args:
        raw: Cleaned raw data.
        events: Events array at current sfreq.
        event_id: MNE event_id dict.

    Returns:
        Preloaded response-locked Epochs, or None if no response events found.
    """
    resp_events: np.ndarray = events[np.isin(events[:, 2], RESP_CODES)]
    resp_event_id: dict[str, int] = {k: v for k, v in event_id.items() if v in RESP_CODES}

    if len(resp_events) == 0:
        print("  No response events found — skipping response-locked epochs")
        return None

    metadata: pd.DataFrame = build_response_metadata(resp_events, raw.info["sfreq"])

    epochs: mne.Epochs = mne.Epochs(
        raw,
        resp_events,
        event_id=resp_event_id,
        tmin=RESP_TMIN,
        tmax=RESP_TMAX,
        baseline=RESP_BASELINE,
        metadata=metadata,
        reject=REJECT_CRIT,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )

    n_kept: int = len(epochs)
    n_total: int = len(resp_events)
    pct: float = 100 * n_kept / n_total if n_total else 0
    print(f"  Response epochs: {n_kept} / {n_total} kept ({pct:.1f} %)")
    for cond in ("Mono", "Di_null", "Di_part", "Di_full"):
        n = (epochs.metadata["condition"] == cond).sum() if epochs.metadata is not None else "?"
        print(f"    {cond}: {n}")

    return epochs


# ─────────────────────────────────────────────────────────────────────────────
# CPP ERP quality-control figure
# ─────────────────────────────────────────────────────────────────────────────


def _cpp_pick_channels(ch_names: list[str]) -> list[str]:
    """Return CPP cluster channels present in this recording."""
    return [c for c in CPP_CHANNELS if c in ch_names]


def _find_cpp_peak(
    evoked: mne.Evoked,
    cpp_ch: list[str],
    t_start: float,
    t_end: float,
) -> float:
    """Return the time of the maximum positive CPP amplitude within [t_start, t_end].

    Args:
        evoked: Averaged evoked response.
        cpp_ch: CPP channel names to average over.
        t_start: Start of the peak search window in seconds.
        t_end: End of the peak search window in seconds.

    Returns:
        Peak latency in seconds.
    """
    picks = mne.pick_channels(evoked.ch_names, include=cpp_ch)
    data = evoked.data[picks].mean(axis=0)
    times = evoked.times
    mask = (times >= t_start) & (times <= t_end)
    if mask.sum() == 0:
        return (t_start + t_end) / 2
    peak_idx: int = int(np.argmax(data[mask]))
    return float(times[mask][peak_idx])


def generate_cpp_qc(
    subject_id: str,
    epochs_stim: mne.Epochs,
    epochs_resp: Optional[mne.Epochs],
    output_dir: Path,
) -> Path:
    """Generate a 2×2 CPP ERP quality-control figure and save to PNG.

    Rows: stimulus-locked (top) and response-locked (bottom).
    Columns: ERP line plot (left) and topomap at peak ±25 ms (right).

    Args:
        subject_id: Subject identifier used in the filename and title.
        epochs_stim: Stimulus-locked epochs.
        epochs_resp: Response-locked epochs, or None if unavailable.
        output_dir: Directory to write the PNG to.

    Returns:
        Path to the saved PNG.
    """
    from scipy.ndimage import gaussian_filter1d

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    configs: list[tuple[Any, str, tuple[float, float], str]] = [
        (epochs_stim, "Stimulus-locked", CPP_STIM_WINDOW, "Time re stimulus onset (s)"),
        (epochs_resp, "Response-locked", CPP_RESP_WINDOW, "Time re response (s)"),
    ]

    for row, (epochs, label, (t0, t1), xlabel) in enumerate(configs):
        if epochs is None or len(epochs) == 0:
            for col in range(2):
                ax = fig.add_subplot(gs[row, col])
                ax.text(
                    0.5, 0.5, f"No {label.lower()} epochs",
                    ha="center", va="center", transform=ax.transAxes, color="gray",
                )
                ax.set_title(f"{label} — unavailable")
            continue

        evoked: mne.Evoked = epochs.average()
        cpp_ch: list[str] = _cpp_pick_channels(evoked.ch_names)
        if not cpp_ch:
            print(f"  Warning: none of {CPP_CHANNELS} found — skipping row {row}")
            continue

        peak_t: float = _find_cpp_peak(evoked, cpp_ch, t0, t1)
        half_w: float = 0.025

        ax_erp = fig.add_subplot(gs[row, 0])
        picks = mne.pick_channels(evoked.ch_names, include=cpp_ch)
        times = evoked.times
        for p in picks:
            ax_erp.plot(times, evoked.data[p] * 1e6, color="#AAAAAA", lw=0.7, alpha=0.6)
        mean_cpp = gaussian_filter1d(evoked.data[picks].mean(axis=0) * 1e6, sigma=2)
        ax_erp.plot(times, mean_cpp, color="#1B4F72", lw=2.0, label="CPP mean")
        ax_erp.axvline(peak_t, color="#C0392B", lw=1.2, ls="--", label=f"Peak {peak_t*1000:.0f} ms")
        ax_erp.axvspan(peak_t - half_w, peak_t + half_w, color="#C0392B", alpha=0.15, label="Topo window")
        ax_erp.axhline(0, color="k", lw=0.5)
        ax_erp.axvline(0, color="k", lw=0.5, ls=":")
        ax_erp.set_xlim(times[0], times[-1])
        ax_erp.set_xlabel(xlabel, fontsize=9)
        ax_erp.set_ylabel("Amplitude (µV)", fontsize=9)
        ax_erp.set_title(f"{label} — CPP cluster (n={len(epochs)})", fontsize=10)
        ax_erp.legend(fontsize=7, loc="upper left")
        ax_erp.tick_params(labelsize=8)

        ax_topo = fig.add_subplot(gs[row, 1])
        try:
            cbar_ax = ax_topo.inset_axes([1.05, 0.1, 0.05, 0.8])
            evoked.plot_topomap(
                times=[peak_t],
                average=half_w * 2,
                axes=[ax_topo, cbar_ax],
                show=False,
                colorbar=True,
                outlines="head",
                sphere="auto",
                sensors=True,
                extrapolate="local",
                border="mean",
            )
            ax_topo.set_title(f"Topomap at {peak_t*1000:.0f} ms (±{half_w*1000:.0f} ms)", fontsize=10)
        except Exception as exc:
            ax_topo.text(
                0.5, 0.5, f"Topomap error:\n{exc}",
                ha="center", va="center", transform=ax_topo.transAxes, fontsize=7, color="red",
            )

    fig.suptitle(f"Subject {subject_id} — CPP ERP Quality Control", fontsize=12)
    out_path: Path = output_dir / f"{subject_id}_cpp_qc.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  CPP QC figure: {out_path.name}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Save checkpoints (Pass 2 only)
# ─────────────────────────────────────────────────────────────────────────────


def save_checkpoints(
    subject_id: str,
    raw: mne.io.Raw,
    ica: mne.preprocessing.ICA,
    epochs_stim: mne.Epochs,
    epochs_resp: Optional[mne.Epochs],
    t_start: datetime,
) -> Path:
    """Save all Pass-2 outputs: cleaned raw, ICA, epochs, metadata, QC figure, log.

    Args:
        subject_id: Subject identifier.
        raw: Final cleaned raw data (ICA applied, bad channels interpolated).
        ica: ICA object with confirmed exclusions.
        epochs_stim: Stimulus-locked epochs.
        epochs_resp: Response-locked epochs, or None.
        t_start: Pipeline start time for elapsed-time logging.

    Returns:
        Path to the subject output directory.
    """
    out: Path = OUTPUT_DIR / subject_id
    out.mkdir(parents=True, exist_ok=True)

    raw_file = out / f"{subject_id}_preprocessed_raw.fif"
    ica_file = out / f"{subject_id}_ica.fif"
    stim_file = out / f"{subject_id}_epochs_stimulus.fif"
    resp_file = out / f"{subject_id}_epochs_response.fif"
    stim_meta_file = out / f"{subject_id}_epoch_metadata_stimulus.csv"
    resp_meta_file = out / f"{subject_id}_epoch_metadata_response.csv"
    log_file = out / f"{subject_id}_processing_log.txt"

    raw.save(str(raw_file), overwrite=True, verbose=False)
    ica.save(str(ica_file), overwrite=True, verbose=False)
    epochs_stim.save(str(stim_file), overwrite=True, verbose=False)
    if epochs_stim.metadata is not None:
        epochs_stim.metadata.to_csv(str(stim_meta_file), index=False)
    if epochs_resp is not None:
        epochs_resp.save(str(resp_file), overwrite=True, verbose=False)
        if epochs_resp.metadata is not None:
            epochs_resp.metadata.to_csv(str(resp_meta_file), index=False)

    qc_file: Path = generate_cpp_qc(subject_id, epochs_stim, epochs_resp, out)

    ns: int = len(epochs_stim)
    nst: int = len(epochs_stim.drop_log)
    nr: int = len(epochs_resp) if epochs_resp is not None else 0
    nrt: int = len(epochs_resp.drop_log) if epochs_resp is not None else 0
    elapsed: float = (datetime.now() - t_start).total_seconds()

    log_lines: list[str] = [
        f"Subject:                {subject_id}",
        f"Processed:              {datetime.now().isoformat(timespec='seconds')}",
        f"Elapsed:                {elapsed:.0f} s",
        "",
        f"Sfreq after resample:   {raw.info['sfreq']} Hz",
        f"HP freq (final pass):   {HP_FREQ} Hz",
        f"N channels:             {len(raw.ch_names)}",
        f"Bad channels:           {raw.info.get('bads', [])}",
        f"ICA excluded:           {ica.exclude}",
        "",
        f"Stimulus epochs total:  {nst}",
        f"Stimulus epochs kept:   {ns}  ({100*ns/nst:.1f} %)" if nst else "Stimulus epochs kept:   0",
        f"Response epochs total:  {nrt}",
        f"Response epochs kept:   {nr}  ({100*nr/nrt:.1f} %)" if nrt else "Response epochs kept:   0",
        "",
        "Files saved:",
        f"  {raw_file.name}",
        f"  {ica_file.name}",
        f"  {stim_file.name}",
        f"  {resp_file.name}" if epochs_resp is not None else "  (no response epoch file)",
        f"  {stim_meta_file.name}",
        f"  {resp_meta_file.name}" if epochs_resp is not None else "",
        f"  {qc_file.name}",
    ]
    log_file.write_text("\n".join(line for line in log_lines) + "\n")

    print(f"\n  Output directory: {out}")
    saved: list[Path] = [raw_file, ica_file, stim_file, log_file, qc_file]
    if epochs_resp is not None:
        saved.append(resp_file)
    for f in saved:
        print(f"    ✓ {f.name}")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stage runners
# ─────────────────────────────────────────────────────────────────────────────


def run_ica_stage(subject_id: str) -> None:
    """Pass 1: preprocess with 1 Hz HP, fit ICA, save for human review.

    Args:
        subject_id: Subject folder name under EEG_DIR.
    """
    t0: datetime = datetime.now()
    hr: str = "─" * 60
    print(f"\n{hr}")
    print(f"PASS 1 (ICA identification) — Subject: {subject_id}")
    print(f"High-pass: {HP_FREQ_ICA} Hz")
    print(hr)

    print("\n[1/11] Loading and preprocessing (1 Hz HP)…")
    raw: mne.io.Raw
    events: np.ndarray
    event_id: dict[str, int]
    raw, events, event_id = _run_preprocessing(subject_id, hp_freq=HP_FREQ_ICA)

    print("\n[2/11] Fitting ICA…")
    ica: mne.preprocessing.ICA = run_ica(raw)

    print("\n[3/11] Saving ICA outputs for review…")
    review_path: Path = save_ica_review(subject_id, ica, bad_channels=raw.info["bads"])

    elapsed: float = (datetime.now() - t0).total_seconds()
    print(f"\n✅  Pass 1 complete — {elapsed:.0f} s")
    print(f"\nNext steps:")
    print(f"  1. Inspect: {OUTPUT_DIR / subject_id / (subject_id + '_ica_components.png')}")
    print(f"  2. Load ICA in MNE for detailed review if needed.")
    print(f"  3. Edit confirmed_exclude in: {review_path}")
    print(f"  4. Set \"reviewed\": true")
    print(f"  5. Run: python run_pipeline.py --subject {subject_id} --stage final")


def run_final_stage(subject_id: str) -> None:
    """Pass 2: load reviewed ICA, preprocess with 0.1 Hz HP, apply ICA, epoch.

    Args:
        subject_id: Subject folder name under EEG_DIR.

    Raises:
        RuntimeError: If the ICA review JSON has not been marked as reviewed.
    """
    t0: datetime = datetime.now()
    hr: str = "─" * 60
    print(f"\n{hr}")
    print(f"PASS 2 (Final preprocessing) — Subject: {subject_id}")
    print(f"High-pass: {HP_FREQ} Hz")
    print(hr)

    print("\n[1/15] Loading ICA review…")
    review: dict[str, Any] = load_ica_review(subject_id)
    ica_path: Path = OUTPUT_DIR / subject_id / review["ica_file"]
    ica: mne.preprocessing.ICA = mne.preprocessing.read_ica(str(ica_path), verbose=False)
    ica.exclude = review["confirmed_exclude"]
    print(f"  Confirmed exclusions: {ica.exclude}")
    print(f"  Notes: {review.get('notes', '')}")

    print("\n[2/15] Loading and preprocessing (0.1 Hz HP)…")
    raw: mne.io.Raw
    events: np.ndarray
    event_id: dict[str, int]
    raw, events, event_id = _run_preprocessing(subject_id, hp_freq=HP_FREQ)

    # Restore bad channels identified in Pass 1 so the same set is interpolated
    pass1_bads: list[str] = review.get("bad_channels", [])
    if pass1_bads:
        raw.info["bads"] = sorted(set(raw.info["bads"] + pass1_bads))
        print(f"  Restored Pass-1 bad channels: {pass1_bads}")

    print("\n[3/15] Applying ICA and interpolating bad channels…")
    raw = apply_ica_and_interpolate(raw, ica)

    print("\n[4/15] Extracting stimulus-locked epochs…")
    epochs_stim: mne.Epochs = extract_epochs(raw, events, event_id)

    print("\n[5/15] Extracting response-locked epochs…")
    epochs_resp: Optional[mne.Epochs] = extract_response_epochs(raw, events, event_id)

    print(f"\n{hr}")
    print("[6/15] Saving checkpoints and generating CPP QC figure…")
    save_checkpoints(subject_id, raw, ica, epochs_stim, epochs_resp, t0)

    elapsed: float = (datetime.now() - t0).total_seconds()
    print(f"\n✅  Pass 2 complete — {elapsed:.0f} s")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Two-pass EEG preprocessing pipeline — Prior RDM Study.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --subject ANA60 --stage ica
  python run_pipeline.py --subject ANA60 --stage final
""",
    )
    parser.add_argument(
        "--subject",
        default="ANA60",
        help="Subject ID matching folder name under EEG_DIR (default: ANA60)",
    )
    parser.add_argument(
        "--stage",
        choices=["ica", "final"],
        required=True,
        help="\"ica\": fit ICA for human review; \"final\": apply reviewed ICA and epoch",
    )
    args = parser.parse_args()

    if args.stage == "ica":
        run_ica_stage(args.subject)
    else:
        run_final_stage(args.subject)


if __name__ == "__main__":
    main()
