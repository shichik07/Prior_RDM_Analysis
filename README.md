# Prior RDM Analysis

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
│   │       ├── run_pipeline.py   # single-subject EEG preprocessing
│   │       └── run_all.py        # batch wrapper
│   ├── transform/
│   │   └── behavior/
│   │       └── preprocess.py     # behavioural data cleaning (Polars)
│   └── output/
│       └── behavior/
│           ├── plots.py          # RT / accuracy figures
│           └── psychometric_plots.py
├── pyproject.toml
└── uv.lock
```

---

## EEG preprocessing

### Pipeline overview (`run_pipeline.py`)

The pipeline runs entirely in-memory (no BIDS layout required) and produces analysis-ready epochs for a single subject. Steps:

| # | Step | Details |
|---|------|---------|
| 1 | Load BrainVision | Patches internal filename mismatch in `.vhdr`; converts to `RawArray` |
| 2 | Channel config | HEOG/VEOG → EOG type; FCz added as zero-data channel (online reference) |
| 3 | Montage | `standard_1020`; unrecognised channels silently ignored |
| 4 | Extract events | `mne.events_from_annotations()` — triggers stored as annotations |
| 5 | Filter | HP 0.1 Hz (FIR zero-phase) + notch 50 / 100 / 150 Hz |
| 6 | Resample | Downsample to 250 Hz (filter applied **before** resample) |
| 7 | Experiment bounds | Pre- and post-experiment segments annotated as `BAD` |
| 8 | Bad channels | Flatline detection (SD < 0.5 µV); visual inspection deferred to PREP-03 |
| 9 | Reference | Average reference (FCz included) |
| 10 | ICA | 20 components, Picard (quasi-Newton), seed = 97; auto EOG detection |
| 11 | Apply ICA | Artefact components removed; bad channels interpolated |
| 12 | Stimulus epochs | Onset codes 31–64, −200 to 1000 ms, baseline −200 to 0 ms |
| 13 | Response epochs | Response codes 131–168, −1000 to 500 ms, baseline −1000 to −800 ms |
| 14 | Epoch rejection | EEG > 100 µV or EOG > 200 µV dropped |
| 15 | Save + QC | `.fif` files, metadata CSVs, CPP ERP quality-control figure |

**Outputs** (written to `data/processed/eeg/<subject>/`):

```
<subject>_preprocessed_raw.fif
<subject>_ica.fif
<subject>_epochs_stimulus.fif
<subject>_epochs_response.fif
<subject>_epoch_metadata_stimulus.csv
<subject>_epoch_metadata_response.csv
<subject>_cpp_qc.png
<subject>_processing_log.txt
```

### Running the pipeline

**Single subject:**
```bash
python src/extract/eeg/run_pipeline.py --subject ANA60
```

**All subjects** (auto-discovers folders in `E:\priorRDM\Study\EEGData`):
```bash
python src/extract/eeg/run_all.py
```

**Specific subjects:**
```bash
python src/extract/eeg/run_all.py --subject ANA60 ANA61 ANA62
```

**Skip already-processed subjects** (safe to resume after interruption):
```bash
python src/extract/eeg/run_all.py --skip-done
```

**Dry-run** (list subjects without processing):
```bash
python src/extract/eeg/run_all.py --dry-run
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

- **Left column**: stimulus-locked ERP (−200 to 1000 ms), peak topomap at 200–1000 ms window
- **Right column**: response-locked ERP (−1000 to 500 ms), peak topomap at −400 to +50 ms
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
uv run black src/
uv run ruff check src/
uv run mypy src/
uv run pytest
```

---

## Contact

Julius Kricheldorff — julius@kricheldorff.de  
Repository: https://github.com/shichik07/Prior_RDM_Analysis
