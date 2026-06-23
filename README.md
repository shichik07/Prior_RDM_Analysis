# Prior RDM Analysis

**Last updated:** 2026-06-23

EEG and behavioural analysis pipeline for *Informed Visual Decisions in Parkinson's Disease* — Carl von Ossietzky Universität Oldenburg.

The study examines whether PD patients fail to incorporate prior directional information into both the starting point *and* the drift rate of evidence accumulation, using a random-dot-motion paradigm with four prior conditions (uninformative, mono, partial, full).

---

## Installation

**Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).**

```bash
git clone https://github.com/shichik07/Prior_RDM_Analysis.git
cd Prior_RDM_Analysis
uv sync
```

> **External drive**: raw EEG data lives on `/mnt/e/priorRDM/Study/EEGData` (WSL) and is never committed to the repository. Processed outputs go to `data/processed/` (also git-ignored).

---

## Project structure

```
Prior_RDM_Analysis/
├── src/
│   ├── extract/
│   │   └── eeg/
│   │       ├── run_pipeline.py   # two-pass EEG preprocessing entry point
│   │       ├── run_all.py        # batch wrapper
│   └── transform/
│       └── behavior/
│           └── preprocess.py     # behavioural data cleaning (Polars)
├── pyproject.toml
└── uv.lock
```

---

## EEG preprocessing

The pipeline uses a **two-pass approach** to obtain clean, low-frequency-drift-free ICA components:

### Pass 1 — ICA identification (`--stage ica`)

Preprocesses with a 1 Hz high-pass filter (removes slow drifts that smear ICA components), fits 20-component Picard ICA, auto-detects EOG artefacts, and saves outputs for human review.

```bash
python src/extract/eeg/run_pipeline.py --subject ANA60 --stage ica
```

**Human review (required before Pass 2):**
1. Inspect `data/processed/eeg/ANA60/ANA60_ica_components.png`.
2. Load `ANA60_ica.fif` in MNE for detailed inspection if needed.
3. Edit `confirmed_exclude` in `ANA60_ica_review.json` if the auto-detection missed or over-flagged components.
4. Set `"reviewed": true` in the JSON.

### Pass 2 — Final preprocessing (`--stage final`)

Reloads the raw data, preprocesses with the ERP-appropriate 0.1 Hz high-pass, applies the human-confirmed ICA components, and extracts epochs.

```bash
python src/extract/eeg/run_pipeline.py --subject ANA60 --stage final
```

### Pipeline steps

| # | Step | Details |
|---|------|---------|
| 1 | Load BrainVision | Patches internal filename mismatch in `.vhdr`; converts to `RawArray` |
| 2 | Channel config | HEOG/VEOG → EOG type; FCz added as zero-data channel (online reference) |
| 3 | Montage | `standard_1020`; unrecognised channels silently ignored |
| 4 | Extract events | `mne.events_from_annotations()` |
| 5 | Filter | **Pass 1**: HP 1 Hz; **Pass 2**: HP 0.1 Hz. Both: notch 50/100/150 Hz. Applied before resample. |
| 6 | Resample | Downsample to 250 Hz |
| 7 | Re-extract events | Sample numbers shift after resample |
| 8 | Experiment bounds | Pre- and post-experiment segments annotated as `BAD` |
| 9 | Bad channels | Flatline detection (SD < 0.5 µV) |
| 10 | Reference | Average reference (FCz included) |
| 11† | ICA | 20 components, Picard (seed=97); auto EOG detection | Pass 1 only |
| 12† | Human review | Edit `*_ica_review.json`, set `"reviewed": true` | Pass 1 only |
| 11‡ | Apply ICA | Confirmed components removed; bad channels interpolated | Pass 2 only |
| 12‡ | Stimulus epochs | Onset codes 31–64, −200 to 1000 ms, baseline −200 to 0 ms | Pass 2 only |
| 13‡ | Response epochs | Response codes 131–168, −1000 to 500 ms, baseline −1000 to −800 ms | Pass 2 only |
| 14‡ | Epoch rejection | EEG > 100 µV or EOG > 200 µV dropped | Pass 2 only |
| 15‡ | Save + QC | `.fif` files, metadata CSVs, CPP ERP QC figure, log | Pass 2 only |

**Outputs** (written to `data/processed/eeg/<subject>/`):

```
<subject>_ica.fif                       — fitted ICA (Pass 1)
<subject>_ica_components.png            — component topomaps for review (Pass 1)
<subject>_ica_review.json               — exclusion list + review flag (Pass 1)
<subject>_preprocessed_raw.fif          — final cleaned continuous data (Pass 2)
<subject>_epochs_stimulus.fif           — stimulus-locked epochs (Pass 2)
<subject>_epochs_response.fif           — response-locked epochs (Pass 2)
<subject>_epoch_metadata_stimulus.csv   — per-trial condition/coherence table (Pass 2)
<subject>_epoch_metadata_response.csv   — per-trial accuracy table (Pass 2)
<subject>_cpp_qc.png                    — CPP ERP + topomap QC figure (Pass 2)
<subject>_processing_log.txt            — summary stats and file list (Pass 2)
```

### Batch processing

```bash
python src/extract/eeg/run_all.py --stage ica              # Pass 1 all subjects
# ... review all subjects ...
python src/extract/eeg/run_all.py --stage final            # Pass 2 all subjects
python src/extract/eeg/run_all.py --stage final --skip-done
```

### Trigger code scheme

Stimulus onsets are encoded as `10 × condition + coherence`:

| Condition | Codes |
|-----------|-------|
| Mono | 31–34 |
| Di null (uninformative) | 41–44 |
| Di partial | 51–54 |
| Di full | 61–64 |

Response triggers (correct / incorrect × 4 conditions × 4 units) span 131–168.

### CPP quality-control figure

`_cpp_qc.png` shows the centro-parietal positivity (CPP) for all accepted epochs:

- **Left column**: ERP at CPP cluster channels, peak marked
- **Right column**: topomap at peak ±25 ms
- **Row 1**: stimulus-locked (peak search 200–1000 ms)
- **Row 2**: response-locked (peak search −400 to +50 ms)
- CPP channels: Pz, CPz, P1, P2, P3, P4, CP1, CP2, CP3, CP4, POz

---

## Behavioural preprocessing

```bash
# Clean and validate raw PsychoPy output
python src/transform/behavior/preprocess.py

# Figures (RT distributions, accuracy, psychometric curves)
python src/output/behavior/plots.py
python src/output/behavior/psychometric_plots.py
```

---

## Code quality

```bash
uv run ruff format src/
uv run ruff check src/
uv run pyright src/
uv run pytest
```

Pre-commit hooks enforce all of the above automatically on each commit. Install once with:

```bash
pre-commit install
```

---

## Contact

Julius Kricheldorff — julius@kricheldorff.de  
Repository: https://github.com/shichik07/Prior_RDM_Analysis

---

## Changelog

### 2026-06-23
- Refactor EEG pipeline into two-pass approach: 1 Hz HP + ICA identification (Pass 1) and 0.1 Hz HP + ICA application + epoching (Pass 2).
- Add human review step via `{subject}_ica_review.json` between passes.
- Add `--stage ica | final` argument to `run_pipeline.py`.
- Update README pipeline table and output file list to reflect two-pass structure.

### 2026-06-21
- Add `Last updated` header and `Changelog` section.
- Replace `black` / `mypy` with `ruff-format` / `pyright` in code quality instructions.
- Add `pre-commit install` step to code quality section.
