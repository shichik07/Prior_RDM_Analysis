#!/usr/bin/env python3
"""
PREP-01: EEG Preprocessing Pipeline — Prior RDM Study
=======================================================
Standalone runner. Bypasses BIDS loading; works directly with raw
BrainVision files and patches the internal filename mismatch that
is present in ANA60 (and potentially other subjects).

Usage:
    python run_pipeline.py --subject ANA60
    python run_pipeline.py            # defaults to ANA60

Pipeline steps:
    1.  Load BrainVision data (filename-patch workaround)
    2.  Configure channels (HEOG/VEOG → EOG type)
    3.  Add FCz as zero-data channel (was online reference)
    4.  Set standard_1020 montage
    5.  Extract events from annotations (mne.events_from_annotations)
    6.  Resample to 250 Hz
    7.  Mark pre- / post-experiment segments as BAD
    8.  Highpass filter 0.1 Hz (FIR, zero-phase)
    9.  Notch filter 50 / 100 / 150 Hz
    10. Detect bad channels (flatline + low-correlation)
    11. Average reference (FCz included)
    12. ICA (20 components, Picard, seed=97); auto EOG detection
    13. Apply ICA + interpolate bad channels
    14. Extract epochs (stimulus onsets, −200 to 1000 ms)
    15. Baseline correction (−200 to 0 ms)
    16. Epoch rejection (EEG > 100 µV, EOG > 200 µV)
    17. Save checkpoints (.fif + metadata CSV + log)

Outputs (data/processed/eeg/<subject>/):
    <subject>_preprocessed_raw.fif
    <subject>_ica.fif
    <subject>_epochs_stimulus.fif       — stimulus-locked (-200 to 1000 ms)
    <subject>_epochs_response.fif       — response-locked (-1000 to 500 ms)
    <subject>_epoch_metadata_stimulus.csv
    <subject>_epoch_metadata_response.csv
    <subject>_cpp_qc.png                — CPP ERP + topomap QC figure
    <subject>_processing_log.txt
"""

import argparse
import re
import shutil
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).resolve().parents[3]   # repo root
EEG_DIR     = Path(r"E:\priorRDM\Study\EEGData")   # raw data on external drive
OUTPUT_DIR  = BASE_DIR / "data" / "processed" / "eeg"

SFREQ_TARGET = 250           # Hz
HP_FREQ      = 0.1           # Hz highpass
NOTCH_FREQS  = [50, 100, 150]
N_ICA        = 20
ICA_SEED     = 97
TMIN, TMAX   = -0.200, 1.000  # epoch window (s)
BASELINE     = (-0.200, 0.0)
REJECT_CRIT  = {"eeg": 100e-6, "eog": 200e-6}

# Trigger codes for stimulus onsets (Con_num × 10 + Coh_num)
# Mono=3x, Di_null=4x, Di_part=5x, Di_full=6x; coherence 1–4
ONSET_CODES = (
    list(range(31, 35)) +   # Mono
    list(range(41, 45)) +   # Di_null
    list(range(51, 55)) +   # Di_part
    list(range(61, 65))     # Di_full
)

BLOCK_START_CODES = list(range(1, 9))    # 1–8: start block
BLOCK_END_CODES   = list(range(11, 19))  # 11–18: end block

# Decode trigger codes → labels
COND_MAP  = {3: "Mono", 4: "Di_null", 5: "Di_part", 6: "Di_full"}
COH_MAP   = {1: 0.0, 2: round(2/30, 4), 3: round(4/30, 4), 4: round(10/30, 4)}
COH_LABEL = {1: "0%", 2: "6.7%", 3: "13.3%", 4: "33.3%"}

# ── Response-locked epoch settings ────────────────────────────────────────────
# Response codes: 100 + Con_num*10 + Coh_num (correct: 1-4, incorrect: 5-8)
RESP_CODES  = list(range(131, 139)) + list(range(141, 149)) + \
              list(range(151, 159)) + list(range(161, 169))
RESP_TMIN, RESP_TMAX = -1.000, 0.500
RESP_BASELINE        = (-1.000, -0.800)   # pre-stimulus window

# ── Centro-parietal positivity (CPP) channel cluster ──────────────────────────
CPP_CHANNELS = ["Pz", "CPz", "P1", "P2", "P3", "P4", "CP1", "CP2",
                "CP3", "CP4", "POz"]
# CPP peak search windows (seconds)
CPP_STIM_WINDOW = (0.200, 1.000)   # stimulus-locked: 200–1000 ms
CPP_RESP_WINDOW = (-0.400, 0.050)  # response-locked:  -400 to +50 ms


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 · Load BrainVision (with filename-patch workaround)
# ─────────────────────────────────────────────────────────────────────────────

def load_brainvision(subject_id: str) -> mne.io.Raw:
    """
    Load a BrainVision recording, patching any internal filename mismatches.

    Some recordings have DataFile/MarkerFile headers pointing to a generic
    name (e.g. prior_rdm_0040.eeg) while the actual files use the subject ID.
    We patch those references in a temp directory before loading.
    """
    subj_dir   = EEG_DIR / subject_id
    vhdr_paths = list(subj_dir.glob("*.vhdr"))
    if not vhdr_paths:
        raise FileNotFoundError(f"No .vhdr found in {subj_dir}")

    vhdr_path = vhdr_paths[0]
    stem      = vhdr_path.stem   # e.g. 'RDM_HC_participant_ANA60'
    vmrk_path = vhdr_path.with_suffix(".vmrk")
    eeg_path  = vhdr_path.with_suffix(".eeg")

    vhdr_text = vhdr_path.read_text(encoding="utf-8", errors="replace")
    vmrk_text = vmrk_path.read_text(encoding="utf-8", errors="replace")

    # Patch DataFile= and MarkerFile= lines
    vhdr_fixed = re.sub(r"(?i)DataFile=.*",   f"DataFile={stem}.eeg",  vhdr_text)
    vhdr_fixed = re.sub(r"(?i)MarkerFile=.*", f"MarkerFile={stem}.vmrk", vhdr_fixed)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / f"{stem}.vhdr").write_text(vhdr_fixed, encoding="utf-8")
        (tmp / f"{stem}.vmrk").write_text(vmrk_text,  encoding="utf-8")
        shutil.copy(eeg_path, tmp / f"{stem}.eeg")

        _raw = mne.io.read_raw_brainvision(
            str(tmp / f"{stem}.vhdr"),
            preload=True,
            verbose=False,
        )
        # Convert to RawArray so data is fully in-memory and
        # no longer references the (now-deleted) temp files.
        raw = mne.io.RawArray(_raw.get_data(), _raw.info, verbose=False)
        raw.set_annotations(_raw.annotations)

    print(f"  Loaded: {len(raw.ch_names)} ch, {raw.info['sfreq']:.0f} Hz, "
          f"{raw.times[-1]/60:.1f} min")
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 · Channel configuration
# ─────────────────────────────────────────────────────────────────────────────

def configure_channels(raw: mne.io.Raw) -> mne.io.Raw:
    eog_chs = [c for c in ("HEOG", "VEOG") if c in raw.ch_names]
    if eog_chs:
        raw.set_channel_types({c: "eog" for c in eog_chs})
        print(f"  EOG channels: {eog_chs}")

    # Add FCz as a flat (zero) channel — it was the online reference
    if "FCz" not in raw.ch_names:
        fcz_info = mne.create_info(["FCz"], raw.info["sfreq"], ch_types="eeg")
        fcz_raw  = mne.io.RawArray(
            np.zeros((1, len(raw.times))), fcz_info, verbose=False
        )
        raw.add_channels([fcz_raw], force_update_info=True)
        print("  Added FCz (zero reference channel)")

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 · Montage
# ─────────────────────────────────────────────────────────────────────────────

def set_montage(raw: mne.io.Raw) -> mne.io.Raw:
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="ignore", verbose=False)
    print("  standard_1020 montage set (unrecognised channels ignored)")
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 · Events from annotations
# ─────────────────────────────────────────────────────────────────────────────

def get_events(raw: mne.io.Raw):
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    unique = sorted(set(events[:, 2].tolist()))
    print(f"  {len(events)} events, {len(unique)} unique codes: {unique}")
    return events, event_id


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 · Resample
# ─────────────────────────────────────────────────────────────────────────────

def resample(raw: mne.io.Raw) -> mne.io.Raw:
    raw.resample(SFREQ_TARGET, npad="auto", verbose=False)
    print(f"  Resampled → {SFREQ_TARGET} Hz")
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 · Mark pre- / post-experiment data as BAD
# ─────────────────────────────────────────────────────────────────────────────

def mark_experiment_bounds(raw: mne.io.Raw, events: np.ndarray) -> mne.io.Raw:
    """
    Annotate data outside the experimental period with BAD_ labels.
    Uses block-start triggers (1–8) or practice-start (200) to find
    the beginning, and the last response trigger to find the end.
    """
    sfreq = raw.info["sfreq"]
    rec_end = raw.times[-1]
    new_anns = []

    # --- Pre-experiment ---
    boundary_codes = BLOCK_START_CODES + [200]
    starts = events[np.isin(events[:, 2], boundary_codes)]
    if len(starts):
        first_t = starts[0, 0] / sfreq
        pre_buffer = 2.0
        if first_t > pre_buffer:
            dur = first_t - pre_buffer
            new_anns.append((0.0, dur, "BAD_pre_experiment"))
            print(f"  BAD_pre_experiment: {dur:.1f} s")

    # --- Post-experiment ---
    resp_codes = list(range(130, 170)) + BLOCK_END_CODES + [201]
    ends = events[np.isin(events[:, 2], resp_codes)]
    if len(ends):
        last_t = ends[-1, 0] / sfreq
        post_buffer = 2.0
        if rec_end > last_t + post_buffer:
            onset = last_t + post_buffer
            dur   = rec_end - onset
            new_anns.append((onset, dur, "BAD_post_experiment"))
            print(f"  BAD_post_experiment: {dur:.1f} s")

    if new_anns:
        onsets      = [a[0] for a in new_anns]
        durations   = [a[1] for a in new_anns]
        descriptions= [a[2] for a in new_anns]
        extra = mne.Annotations(onsets, durations, descriptions,
                                orig_time=raw.annotations.orig_time)
        raw.set_annotations(raw.annotations + extra)

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 · Filtering
# ─────────────────────────────────────────────────────────────────────────────

def apply_filters(raw: mne.io.Raw) -> mne.io.Raw:
    raw.filter(
        l_freq=HP_FREQ, h_freq=None,
        method="fir", fir_design="firwin", phase="zero",
        verbose=False,
    )
    print(f"  HP filter: {HP_FREQ} Hz (FIR, zero-phase)")

    raw.notch_filter(NOTCH_FREQS, method="fir", verbose=False)
    print(f"  Notch filter: {NOTCH_FREQS} Hz")

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 · Bad channel detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_bad_channels(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Automatic bad channel detection — flatline criterion only.

    Flatline: channel SD < 0.5 uV (effectively no signal).
    FCz is excluded since it is flat by design (online reference).

    Note: visual inspection and neighbourhood-correlation-based detection
    happen in PREP-03 (manual ICA and QC step).
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, exclude=[])
    eeg_chs   = [raw.ch_names[i] for i in eeg_picks]
    chs_check = [c for c in eeg_chs if c != "FCz"]
    picks     = [raw.ch_names.index(c) for c in chs_check]

    if not picks:
        return raw

    data = raw.get_data(picks=picks)

    # Flatline: SD < 0.5 uV
    std      = np.std(data, axis=1)
    flatline = [chs_check[i] for i, s in enumerate(std) if s < 0.5e-6]

    if flatline:
        raw.info["bads"] = sorted(set(raw.info["bads"] + flatline))
        print(f"  Bad channels (flatline): {flatline}")
    else:
        print("  No flatline channels detected")
        print("  (Visual/neighbourhood inspection deferred to PREP-03)")

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Step 9 · Average reference
# ─────────────────────────────────────────────────────────────────────────────

def set_reference(raw: mne.io.Raw) -> mne.io.Raw:
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
    print("  Average reference applied (FCz included)")
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Step 10 · ICA
# ─────────────────────────────────────────────────────────────────────────────

def run_ica(raw: mne.io.Raw) -> mne.preprocessing.ICA:
    ica = mne.preprocessing.ICA(
        n_components=N_ICA,
        method="picard",
        random_state=ICA_SEED,
        max_iter=1000,
        verbose=False,
    )
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, exclude="bads")
    ica.fit(raw, picks=eeg_picks, verbose=False)
    print(f"  ICA fitted: {N_ICA} components (Picard, seed={ICA_SEED})")

    # Automatic EOG artifact detection
    try:
        eog_idx, _ = ica.find_bads_eog(raw, verbose=False)
        ica.exclude = eog_idx
        print(f"  Auto-detected EOG components: {eog_idx}")
        print("  ⚠  Manual review required in PREP-03 before analysis")
    except Exception as exc:
        print(f"  EOG auto-detection skipped: {exc}")

    return ica


# ─────────────────────────────────────────────────────────────────────────────
# Step 11 · Apply ICA + interpolate bad channels
# ─────────────────────────────────────────────────────────────────────────────

def apply_ica_and_interpolate(raw: mne.io.Raw, ica: mne.preprocessing.ICA) -> mne.io.Raw:
    if ica.exclude:
        ica.apply(raw, verbose=False)
        print(f"  ICA applied: removed components {ica.exclude}")
    else:
        print("  ICA: no components removed (PREP-03 manual review pending)")

    if raw.info["bads"]:
        bads_before = raw.info["bads"][:]
        raw.interpolate_bads(reset_bads=True, verbose=False)
        print(f"  Interpolated: {bads_before}")

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Step 12 · Epoch extraction
# ─────────────────────────────────────────────────────────────────────────────

def build_trial_metadata(onset_events: np.ndarray, sfreq: float) -> pd.DataFrame:
    """Decode trigger codes into a per-trial metadata table."""
    rows = []
    for ev in onset_events:
        code = int(ev[2])
        cd   = code // 10
        chd  = code % 10
        rows.append({
            "onset_code":      code,
            "condition":       COND_MAP.get(cd,  "Unknown"),
            "coherence":       COH_MAP.get(chd,  np.nan),
            "coherence_label": COH_LABEL.get(chd, "Unknown"),
            "onset_sample":    int(ev[0]),
            "onset_time_s":    round(ev[0] / sfreq, 4),
        })
    return pd.DataFrame(rows)


def extract_epochs(raw: mne.io.Raw, events: np.ndarray, event_id: dict) -> mne.Epochs:
    onset_events   = events[np.isin(events[:, 2], ONSET_CODES)]
    onset_event_id = {k: v for k, v in event_id.items() if v in ONSET_CODES}

    metadata = build_trial_metadata(onset_events, raw.info["sfreq"])

    epochs = mne.Epochs(
        raw,
        onset_events,
        event_id=onset_event_id,
        tmin=TMIN, tmax=TMAX,
        baseline=BASELINE,
        metadata=metadata,
        reject=REJECT_CRIT,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )

    n_kept  = len(epochs)
    n_total = len(onset_events)
    pct     = 100 * n_kept / n_total if n_total else 0
    print(f"  Epochs: {n_kept} / {n_total} kept ({pct:.1f} %)")

    # Condition breakdown
    for cond in ("Mono", "Di_null", "Di_part", "Di_full"):
        n = (epochs.metadata["condition"] == cond).sum() if epochs.metadata is not None else "?"
        print(f"    {cond}: {n}")

    return epochs



# ─────────────────────────────────────────────────────────────────────────────
# Response-locked epoch extraction
# ─────────────────────────────────────────────────────────────────────────────

def build_response_metadata(resp_events: np.ndarray, sfreq: float) -> pd.DataFrame:
    """Decode response trigger codes into per-trial metadata."""
    RESP_COND_MAP = {3: "Mono", 4: "Di_null", 5: "Di_part", 6: "Di_full"}
    rows = []
    for ev in resp_events:
        code  = int(ev[2])
        cd    = (code - 100) // 10          # condition digit (3-6)
        units = code % 10                   # 1-4 correct, 5-8 incorrect
        if 1 <= units <= 4:
            accuracy, coh_num = "correct",   units
        elif 5 <= units <= 8:
            accuracy, coh_num = "incorrect", units - 4
        else:
            accuracy, coh_num = "other",     0
        rows.append({
            "resp_code":       code,
            "condition":       RESP_COND_MAP.get(cd, "Unknown"),
            "coherence":       COH_MAP.get(coh_num, np.nan),
            "coherence_label": COH_LABEL.get(coh_num, "Unknown"),
            "accuracy":        accuracy,
            "resp_sample":     int(ev[0]),
            "resp_time_s":     round(ev[0] / sfreq, 4),
        })
    return pd.DataFrame(rows)


def extract_response_epochs(raw: mne.io.Raw,
                             events: np.ndarray,
                             event_id: dict) -> mne.Epochs:
    """
    Extract response-locked epochs.
    Window: -1000 to +500 ms around each response trigger.
    Baseline: -1000 to -800 ms (well before the response).
    """
    resp_events   = events[np.isin(events[:, 2], RESP_CODES)]
    resp_event_id = {k: v for k, v in event_id.items() if v in RESP_CODES}

    if len(resp_events) == 0:
        print("  No response events found in this segment — skipping")
        return None

    metadata = build_response_metadata(resp_events, raw.info["sfreq"])

    epochs = mne.Epochs(
        raw,
        resp_events,
        event_id=resp_event_id,
        tmin=RESP_TMIN, tmax=RESP_TMAX,
        baseline=RESP_BASELINE,
        metadata=metadata,
        reject=REJECT_CRIT,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )

    n_kept  = len(epochs)
    n_total = len(resp_events)
    pct     = 100 * n_kept / n_total if n_total else 0
    print(f"  Response epochs: {n_kept} / {n_total} kept ({pct:.1f} %)")
    for cond in ("Mono", "Di_null", "Di_part", "Di_full"):
        n = (epochs.metadata["condition"] == cond).sum() if epochs.metadata is not None else "?"
        print(f"    {cond}: {n}")

    return epochs


# ─────────────────────────────────────────────────────────────────────────────
# CPP ERP quality-control figure
# ─────────────────────────────────────────────────────────────────────────────

def _cpp_pick_channels(ch_names: list) -> list:
    """Return CPP cluster channels that are present in this recording."""
    return [c for c in CPP_CHANNELS if c in ch_names]


def _find_cpp_peak(evoked: mne.Evoked, cpp_ch: list,
                   t_start: float, t_end: float) -> float:
    """Return the time of the positive CPP peak within [t_start, t_end]."""
    picks = mne.pick_channels(evoked.ch_names, include=cpp_ch)
    data  = evoked.data[picks].mean(axis=0)          # mean across CPP cluster
    times = evoked.times
    mask  = (times >= t_start) & (times <= t_end)
    if mask.sum() == 0:
        return (t_start + t_end) / 2
    peak_idx = np.argmax(data[mask])
    return float(times[mask][peak_idx])


def generate_cpp_qc(subject_id: str,
                    epochs_stim: mne.Epochs,
                    epochs_resp: mne.Epochs,
                    output_dir: Path) -> Path:
    """
    Generate a 2x2 QC figure for the centro-parietal positivity (CPP):
      Row 1 — Stimulus-locked ERP (200–1000 ms peak search)
      Row 2 — Response-locked  ERP (-400 to +50 ms peak search)
    Columns: (a) ERP line plot at CPP channels + mean; (b) topomap ±25 ms at peak.
    """
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    configs = [
        (epochs_stim, "Stimulus-locked", CPP_STIM_WINDOW, "Time re stimulus onset (s)"),
        (epochs_resp, "Response-locked",  CPP_RESP_WINDOW, "Time re response (s)"),
    ]

    for row, (epochs, label, (t0, t1), xlabel) in enumerate(configs):
        if epochs is None or len(epochs) == 0:
            for col in range(2):
                ax = fig.add_subplot(gs[row, col])
                ax.text(0.5, 0.5, f"No {label.lower()} epochs",
                        ha="center", va="center", transform=ax.transAxes, color="gray")
                ax.set_title(f"{label} — unavailable")
            continue

        evoked  = epochs.average()
        cpp_ch  = _cpp_pick_channels(evoked.ch_names)
        if not cpp_ch:
            print(f"  Warning: none of {CPP_CHANNELS} found — skipping topomap row {row}")
            continue

        peak_t  = _find_cpp_peak(evoked, cpp_ch, t0, t1)
        half_w  = 0.025          # ±25 ms = 50 ms window

        # ── Column A: ERP line plot ───────────────────────────────────────────
        ax_erp = fig.add_subplot(gs[row, 0])
        picks  = mne.pick_channels(evoked.ch_names, include=cpp_ch)
        times  = evoked.times
        for p in picks:
            ax_erp.plot(times, evoked.data[p] * 1e6,
                        color="#AAAAAA", lw=0.7, alpha=0.6)
        from scipy.ndimage import gaussian_filter1d
        mean_cpp_raw = evoked.data[picks].mean(axis=0) * 1e6
        mean_cpp     = gaussian_filter1d(mean_cpp_raw, sigma=2)  # ~8 ms smoothing
        ax_erp.plot(times, mean_cpp, color="#1B4F72", lw=2.0, label="CPP mean")
        ax_erp.axvline(peak_t, color="#C0392B", lw=1.2, ls="--",
                       label=f"Peak {peak_t*1000:.0f} ms")
        ax_erp.axvspan(peak_t - half_w, peak_t + half_w,
                       color="#C0392B", alpha=0.15, label="Topo window")
        ax_erp.axhline(0, color="k", lw=0.5)
        ax_erp.axvline(0, color="k", lw=0.5, ls=":")
        ax_erp.set_xlim(times[0], times[-1])
        ax_erp.set_xlabel(xlabel, fontsize=9)
        ax_erp.set_ylabel("Amplitude (µV)", fontsize=9)
        ax_erp.set_title(f"{label} — CPP cluster (n={len(epochs)})", fontsize=10)
        ax_erp.legend(fontsize=7, loc="upper left")
        ax_erp.tick_params(labelsize=8)

        # ── Column B: Topomap at peak ±25 ms ─────────────────────────────────
        ax_topo = fig.add_subplot(gs[row, 1])
        try:
            # plot_topomap with colorbar=True needs 2 axes: map + colorbar
            cbar_ax = ax_topo.inset_axes([1.05, 0.1, 0.05, 0.8])
            evoked.plot_topomap(
                times=[peak_t],
                average=half_w * 2,     # 50 ms window
                axes=[ax_topo, cbar_ax],
                show=False,
                colorbar=True,
                outlines="head",
                sphere="auto",
                sensors=True,
                extrapolate="local",    # no fill beyond electrode positions
                border="mean",          # neutral border value
            )
            ax_topo.set_title(
                f"Topomap at {peak_t*1000:.0f} ms (±{half_w*1000:.0f} ms)",
                fontsize=10
            )
        except Exception as exc:
            ax_topo.text(0.5, 0.5, f"Topomap error:\n{exc}",
                         ha="center", va="center", transform=ax_topo.transAxes,
                         fontsize=7, color="red")

    fig.suptitle(f"Subject {subject_id} — CPP ERP Quality Control", fontsize=12)

    out_path = output_dir / f"{subject_id}_cpp_qc.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  CPP QC figure: {out_path.name}")
    return out_path

# ─────────────────────────────────────────────────────────────────────────────
# Save checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoints(
    subject_id: str,
    raw: mne.io.Raw,
    ica: mne.preprocessing.ICA,
    epochs_stim: mne.Epochs,
    epochs_resp: mne.Epochs,
    t_start: datetime,
) -> Path:
    out = OUTPUT_DIR / subject_id
    out.mkdir(parents=True, exist_ok=True)

    raw_file       = out / f"{subject_id}_preprocessed_raw.fif"
    ica_file       = out / f"{subject_id}_ica.fif"
    stim_file      = out / f"{subject_id}_epochs_stimulus.fif"
    resp_file      = out / f"{subject_id}_epochs_response.fif"
    stim_meta_file = out / f"{subject_id}_epoch_metadata_stimulus.csv"
    resp_meta_file = out / f"{subject_id}_epoch_metadata_response.csv"
    log_file       = out / f"{subject_id}_processing_log.txt"

    raw.save(str(raw_file),  overwrite=True, verbose=False)
    ica.save(str(ica_file),  overwrite=True, verbose=False)
    epochs_stim.save(str(stim_file), overwrite=True, verbose=False)
    if epochs_stim.metadata is not None:
        epochs_stim.metadata.to_csv(str(stim_meta_file), index=False)
    if epochs_resp is not None:
        epochs_resp.save(str(resp_file), overwrite=True, verbose=False)
        if epochs_resp.metadata is not None:
            epochs_resp.metadata.to_csv(str(resp_meta_file), index=False)

    # QC figure
    qc_file = generate_cpp_qc(subject_id, epochs_stim, epochs_resp, out)

    # Epoch stats
    ns  = len(epochs_stim)
    nst = len(epochs_stim.drop_log)
    nr  = len(epochs_resp) if epochs_resp is not None else 0
    nrt = len(epochs_resp.drop_log) if epochs_resp is not None else 0

    elapsed = (datetime.now() - t_start).total_seconds()
    log_lines = [
        f"Subject:                {subject_id}",
        f"Processed:              {datetime.now().isoformat(timespec='seconds')}",
        f"Elapsed:                {elapsed:.0f} s",
        "",
        f"Sfreq after resample:   {raw.info['sfreq']} Hz",
        f"N channels:             {len(raw.ch_names)}",
        f"Bad channels:           {raw.info['bads']}",
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
        f"  {qc_file.name}" if qc_file else "",
    ]
    log_file.write_text("\n".join(l for l in log_lines) + "\n")

    print(f"\n  Output directory: {out}")
    saved = [raw_file, ica_file, stim_file, log_file]
    if epochs_resp is not None: saved.append(resp_file)
    if qc_file: saved.append(qc_file)
    for f in saved:
        print(f"    ✓ {f.name}")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(subject_id: str) -> None:
    t0 = datetime.now()
    hr = "─" * 60

    print(f"\n{hr}")
    print(f"PREP-01  EEG Preprocessing — Subject: {subject_id}")
    print(f"{hr}")

    # 1. Load
    print("\n[1/13] Loading data…")
    raw = load_brainvision(subject_id)

    # 2. Channel config
    print("\n[2/13] Configuring channels…")
    raw = configure_channels(raw)

    # 3. Montage
    print("\n[3/13] Setting montage…")
    raw = set_montage(raw)

    # 4. Events (at original sfreq, before resample)
    print("\n[4/13] Extracting events from annotations…")
    events_orig, event_id = get_events(raw)

    # 5. Filter BEFORE resampling (notch at 150 Hz requires sfreq > 300 Hz)
    print("\n[5/13] Filtering (before resample)…")
    raw = apply_filters(raw)

    # 6. Resample
    print("\n[6/13] Resampling…")
    raw = resample(raw)

    # Re-extract events at new sfreq (sample numbers changed)
    print("\n[7/13] Re-extracting events after resample…")
    events, event_id = get_events(raw)

    # 7. Mark experiment bounds
    print("\n[8/13] Marking pre/post-experiment segments as BAD…")
    raw = mark_experiment_bounds(raw, events)

    # 8. Bad channel detection
    print("\n[9/13] Detecting bad channels…")
    raw = detect_bad_channels(raw)

    # 9. Average reference
    print("\n[10/13] Setting average reference…")
    raw = set_reference(raw)

    # 10. ICA
    print("\n[11/13] Running ICA…")
    ica = run_ica(raw)

    # 11. Apply ICA + interpolate
    print("\n[12/13] Applying ICA and interpolating bad channels…")
    raw = apply_ica_and_interpolate(raw, ica)

    # 12. Stimulus-locked epochs
    print("\n[13/15] Extracting stimulus-locked epochs…")
    epochs_stim = extract_epochs(raw, events, event_id)

    # 13. Response-locked epochs
    print("\n[14/15] Extracting response-locked epochs…")
    epochs_resp = extract_response_epochs(raw, events, event_id)

    # Save + QC figure
    print(f"\n{hr}")
    print("[15/15] Saving checkpoints and generating CPP QC figure…")
    save_checkpoints(subject_id, raw, ica, epochs_stim, epochs_resp, t0)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n✅  PREP-01 complete — {elapsed:.0f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PREP-01: Run EEG preprocessing pipeline on one subject."
    )
    parser.add_argument(
        "--subject", default="ANA60",
        help="Subject ID matching folder name under data/eeg/ (default: ANA60)"
    )
    args = parser.parse_args()
    run_pipeline(args.subject)
