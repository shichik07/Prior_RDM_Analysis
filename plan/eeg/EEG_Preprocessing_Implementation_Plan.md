# EEG Preprocessing Implementation Plan
## MATLAB to Python Migration using MNE

### Overview
This document outlines the implementation plan for migrating the MATLAB EEG preprocessing pipeline (EEGLAB-based) to Python using the MNE package. The original pipeline processes adaptive control experiment data through multiple stages including event coding, filtering, artifact removal, ICA, and epoching.

### Original MATLAB Pipeline Summary

#### 1. Data Loading & Setup
- Load EEG data from BIDS format using `pop_loadbv()` (BrainVision format)
- Directory structure setup for input/output
- EEGLAB environment initialization

#### 2. Channel Configuration
- Edit channel labels: channels 1-127 as EEG, 128-129 as EOG
- Remove EOG channels (HEOG, VEOG)
- Add FCz channel with specific coordinates
- Set FCz as reference

#### 3. Event Coding & Annotation
Complex event recoding system based on trigger codes (SXXX format):
- **Part**: Onset, Response, Fixation, Start/End Block
- **AnalyseType**: MI, MC, main_incon, main_con
- **Congruency**: congruent, incongruent
- **Trial**: inducer, diagnostic
- **Answer**: correct, incorrect
- Handle incorrect response events by renaming preceding onset events

#### 4. Temporal Processing
- Resample to 250 Hz
- Data rejection outside experimental blocks
- Remove pre-experiment and post-experiment data

#### 5. Filtering & Basic Cleaning
- Highpass filter at 0.1 Hz
- Detrend data
- Line noise removal at 50 Hz using `pop_cleanline()`

#### 6. Channel Quality Control
- Automatic channel rejection using `pop_clean_rawdata()`:
  - Flatline criterion: 5 seconds
  - Channel criterion: 0.8 correlation threshold
  - Line noise criterion: 5
- Store rejected channel indices for later interpolation

#### 7. Referencing
- Average reference with FCz

#### 8. Advanced Artifact Removal
- ASR (Artifact Subspace Reconstruction) for burst noise removal
- ICA decomposition using RUNICA algorithm
- Manual component rejection (visual inspection)

#### 9. Final Processing
- Interpolate previously rejected channels
- Epoch extraction around stimulus onset events (-0.2 to 1.0 seconds)
- Save intermediate and final datasets

---

## Python Implementation Plan using MNE

### Phase 1: Setup & Dependencies

```python
# Core packages
import mne
from mne_bids import BIDSPath, read_raw_bids
import numpy as np
import pandas as pd

# Additional packages for artifact handling
import autoreject
from mne.preprocessing import ICA
```

### Phase 2: Data Loading Function

**Implementation Location**: `src/extract/loading.py`

**Key Functions**:
- `load_eeg_data(subject_id, bids_root, task='adaptivecontrol')`
- `setup_bids_paths(subject_id, bids_root)`
- `validate_bids_structure(bids_root)`

**Concept**: Load EEG data from BIDS format using MNE-BIDS, handle file validation, and return standardized Raw objects.

### Phase 3: Event Processing Function

**Implementation Location**: `src/extract/events.py`

**Key Functions**:
- `code_events(raw)` - Main event coding function
- `parse_sxxx_code(event_code)` - Parse SXXX trigger codes
- `create_event_metadata(events)` - Create metadata DataFrame
- `handle_incorrect_responses(events)` - Rename preceding onset events

**Concept**: Translate the complex MATLAB SXXX coding system to Python, handling event categorization (part, analysetype, congruency, trial, answer) and managing incorrect response event renaming logic.

### Phase 4: Main Preprocessing Pipeline

**Implementation Location**: `src/extract/preprocessing.py`

**Key Functions**:
- `preprocess_pipeline(raw, metadata)` - Main preprocessing chain
- `configure_channels(raw)` - Channel setup and editing
- `add_fcz_channel(raw)` - Add FCz with proper coordinates
- `reject_experimental_blocks(raw, metadata)` - Remove non-experimental data
- `apply_filtering(raw)` - Highpass, detrending, line noise removal
- `setup_referencing(raw)` - FCz average referencing

**Concept**: Implement the core preprocessing steps including channel configuration, resampling, temporal data rejection, filtering, and referencing, maintaining the same order and parameters as the MATLAB pipeline.

### Phase 5: Advanced Artifact Handling

**Implementation Location**: `src/extract/artifacts.py`

**Key Functions**:
- `detect_bad_channels(raw)` - Automatic channel detection
- `asr_replacement(raw)` - ASR-like functionality using autoreject
- `ica_pipeline(raw)` - ICA decomposition and component removal
- `clean_rawdata_pipeline(raw)` - MNE equivalent of pop_clean_rawdata
- `apply_line_noise_removal(raw)` - 50Hz line noise removal

**Concept**: Implement advanced artifact removal including automatic bad channel detection, ASR replacement using autoreject, ICA decomposition with component selection, and line noise removal. Handle the transition from MATLAB's clean_rawdata to MNE equivalents.

### Phase 6: Final Processing

**Implementation Location**: `src/extract/epoching.py`

**Key Functions**:
- `final_processing(raw, ica, metadata)` - Complete final processing chain
- `apply_ica_and_interpolate(raw, ica)` - Apply ICA and interpolate bad channels
- `create_event_dictionary()` - Map event codes to meaningful names
- `extract_epochs(raw, events, metadata)` - Epoch extraction with metadata
- `save_checkpoints(raw, epochs, subject_id, output_dir)` - Save intermediate results

**Concept**: Complete the pipeline with ICA application, channel interpolation, epoch extraction around stimulus onset events (-0.2 to 1.0 seconds), and implement a checkpoint system for saving intermediate results.

### Phase 7: Checkpoint System

**Implementation Location**: `src/extract/utils.py`

**Key Functions**:
- `save_checkpoints(raw, epochs, subject_id, output_dir)` - Main checkpoint function
- `load_checkpoint(subject_id, stage, output_dir)` - Load saved intermediate data
- `create_output_structure(output_dir, subject_id)` - Setup output directories
- `log_processing_stage(subject_id, stage, timestamp)` - Processing log

**Concept**: Implement a robust checkpoint system that saves intermediate results at key stages (raw preprocessing, ICA, epochs) for debugging, quality control, and resuming interrupted processing.

---

## Implementation Considerations

### 1. ASR Replacement Strategy
- **Challenge**: MNE doesn't have direct ASR implementation
- **Solution**: Use `autoreject` package or custom amplitude thresholding
- **Alternative**: Implement ASR algorithm independently using NumPy/SciPy

### 2. Event Metadata Preservation
- **Approach**: Use MNE's `metadata` parameter in Epochs
- **Implementation**: Pandas DataFrame for complex event properties
- **Benefit**: Maintains rich event information throughout pipeline

### 3. Channel Interpolation
- **Method**: Use `raw.info['bads']` for bad channel marking
- **Function**: `raw.interpolate_bads()` for spherical interpolation
- **Quality Control**: Visual inspection before/after interpolation

### 4. Quality Control Integration
- **ICA Inspection**: Interactive plots for component selection
- **Signal Quality**: Automated metrics + visual verification
- **Checkpoint System**: Save intermediate results for debugging

### 5. Performance Optimization
- **Memory Management**: Use `preload=True` strategically
- **Parallel Processing**: Leverage MNE's n_jobs parameter
- **Batch Processing**: Process multiple subjects efficiently

---

## File Structure Organization

```
eeg_preprocessing/
├── src/
│   ├── __init__.py
│   ├── loading.py          # Data loading functions
│   ├── preprocessing.py    # Main preprocessing pipeline
│   ├── events.py          # Event coding and handling
│   ├── artifacts.py       # Artifact detection and removal
│   ├── ica.py            # ICA-related functions
│   └── utils.py          # Utility functions
├── config/
│   ├── channel_config.yaml
│   ├── filtering_params.yaml
│   └── event_mapping.yaml
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   └── 03_quality_control.ipynb
└── tests/
    ├── test_loading.py
    ├── test_preprocessing.py
    └── test_events.py
```

---

## Next Steps

1. **Setup Development Environment**
   - Install required packages (MNE, MNE-BIDS, autoreject)
   - Configure BIDS directory structure

2. **Implement Core Functions**
   - Start with data loading and basic preprocessing
   - Implement event coding system
   - Add artifact removal steps

3. **Quality Control Framework**
   - Create visualization tools
   - Implement automated quality metrics
   - Set up manual inspection procedures

4. **Testing and Validation**
   - Compare results with MATLAB pipeline
   - Validate on subset of subjects
   - Performance benchmarking

5. **Documentation and Deployment**
   - Complete API documentation
   - Create usage examples
   - Set up continuous integration

---

## Dependencies

```txt
mne>=1.0.0
mne-bids>=0.10.0
autoreject>=0.3.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
scipy>=1.7.0
pyyaml>=6.0
```

This implementation plan provides a comprehensive roadmap for migrating the MATLAB EEG preprocessing pipeline to Python while maintaining the original functionality and leveraging modern Python neuroimaging tools.
