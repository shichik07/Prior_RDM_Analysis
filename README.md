# Prior RDM Analysis

A comprehensive Python implementation of EEG preprocessing for decision-making experiments in Parkinson's disease research, migrating from MATLAB/EEGLAB to modern Python/MNE-Python with polars for efficient data processing.

## Project Summary

This project investigates decision-making impairments in Parkinson's disease (PD) using drift diffusion modeling (DDM) and EEG analysis. Building on Perugini et al.'s discovery that PD patients cannot incorporate prior information into decisions, we examine whether this deficit extends beyond starting point adjustments to drift rate adaptations. Our study tests if informing participants about decision-relevant information (rather than outcome probabilities) reveals different patterns of impairment in PD versus healthy controls.

The comprehensive EEG preprocessing pipeline ensures high-quality neural data for DDM analysis, implementing advanced artifact removal, ICA decomposition, and BIDS-compatible processing. Originally developed in MATLAB/EEGLAB, this Python implementation leverages modern neuroimaging tools (MNE-Python) and efficient data processing (polars) to support rigorous investigation of prior information processing deficits in PD. The pipeline preserves decision-related neural signals while removing noise, enabling precise measurement of cognitive mechanisms underlying impaired decision-making in neurological disorders.

This project provides a complete preprocessing pipeline for EEG data from decision-making experiments in Parkinson's disease research, originally implemented in MATLAB using EEGLAB. The Python implementation maintains all original functionality while leveraging modern neuroimaging tools and efficient data processing with polars.

## Features

- **Complete EEG Preprocessing Pipeline**: Full migration from MATLAB EEGLAB to Python MNE-Python
- **Complex Event Coding System**: Implements the SXXX trigger code parsing for experimental conditions
- **Advanced Artifact Removal**: ASR replacement using autoreject, ICA decomposition, and channel interpolation
- **BIDS Compatibility**: Full support for BIDS-formatted EEG data
- **Modern Python Stack**: Python 3.12, uv package manager, polars for data processing
- **Modular Architecture**: Clean separation of concerns across 6 specialized modules
- **Checkpoint System**: Robust intermediate result saving and loading
- **Command-Line Interface**: Easy batch processing and single-subject analysis
- **Quality Control**: Comprehensive logging and validation throughout the pipeline

## Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Quick Start with uv

```bash
# Clone the repository
git clone https://github.com/shichik07/Prior_RDM_Analysis.git
cd Prior_RDM_Analysis

# Install dependencies with uv
uv sync

# Install development dependencies (optional)
uv sync --extra dev

# Install full environment with docs
uv sync --extra full
```

### Alternative: pip Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Project Structure

```
Prior_RDM_Analysis/
├── src/
│   └── extract/
│       ├── __init__.py              # Package initialization
│       ├── loading.py               # BIDS data loading and validation
│       ├── events.py                # SXXX event coding system
│       ├── preprocessing.py         # Core preprocessing pipeline
│       ├── artifacts.py             # Artifact removal and ICA
│       ├── epoching.py              # Final processing and epoching
│       ├── utils.py                 # Utilities and checkpoint system
│       ├── main_pipeline.py         # Command-line interface
│       └── test_pipeline.py         # Testing framework
├── plan/
│   └── eeg/
│       └── EEG_Preprocessing_Implementation_Plan.md
├── pyproject.toml                   # Project configuration
└── README.md
```

## Usage

### Command Line Interface

Process a single subject:
```bash
uv run eeg-pipeline --bids-root /path/to/bids --output /path/to/output --subject sub-01
```

Process all subjects:
```bash
uv run eeg-pipeline --bids-root /path/to/bids --output /path/to/output
```

Process specific subjects:
```bash
uv run eeg-pipeline --bids-root /path/to/bids --output /path/to/output --subjects sub-01 sub-02 sub-03
```

Create configuration file:
```bash
uv run eeg-pipeline --create-config /path/to/config.json
```

### Python API

```python
from src.extract import (
    load_eeg_data, 
    code_events, 
    preprocess_pipeline,
    detect_bad_channels,
    ica_pipeline,
    final_processing
)
from src.extract.artifacts import apply_line_noise_removal, asr_replacement

# Load data
raw = load_eeg_data('sub-01', '/path/to/bids')

# Process events
events, metadata = code_events(raw)

# Preprocess
raw = preprocess_pipeline(raw, metadata)

# Artifact removal
raw, rejected_info = detect_bad_channels(raw)
raw = apply_line_noise_removal(raw)
raw = asr_replacement(raw)

# ICA
ica = ica_pipeline(raw)

# Final processing
epochs = final_processing(raw, ica, metadata, 'sub-01', '/path/to/output')
```

## Pipeline Stages

### 1. Data Loading
- Load BIDS-formatted EEG data using MNE-BIDS
- Validate directory structure and file integrity
- Support for BrainVision (.vhdr) and other formats

### 2. Event Processing
- Complex SXXX trigger code parsing
- Experimental condition categorization:
  - **Part**: Onset, Response, Fixation, Start/End Block
  - **AnalyseType**: MI, MC, main_incon, main_con
  - **Congruency**: congruent, incongruent
  - **Trial**: inducer, diagnostic
  - **Answer**: correct, incorrect
- Incorrect response event renaming

### 3. Preprocessing
- Channel configuration (EEG/EOG types, FCz addition)
- Resampling to 250 Hz
- Temporal data rejection (outside experimental blocks)
- Highpass filtering (0.1 Hz)
- Detrending
- FCz average referencing

### 4. Artifact Removal
- Automatic bad channel detection
- Line noise removal (50 Hz)
- ASR-like burst rejection using autoreject
- ICA decomposition with component selection
- Channel interpolation

### 5. Epoching
- Epoch extraction (-0.2 to 1.0 seconds around stimuli)
- Metadata preservation
- Quality control metrics
- Evoked response computation

## Configuration

The pipeline supports extensive configuration through JSON files:

```json
{
  "preprocessing": {
    "resample_freq": 250,
    "highpass_freq": 0.1,
    "line_noise_freq": 50,
    "reference_channel": "FCz"
  },
  "artifacts": {
    "flatline_threshold": 1e-6,
    "correlation_threshold": 0.8,
    "line_noise_threshold": 5,
    "n_components": 20
  },
  "epoching": {
    "tmin": -0.2,
    "tmax": 1.0,
    "baseline": null
  }
}
```

## Data Processing with Polars

While MNE-Python handles the neuroimaging data, polars is used for efficient processing of:
- Event metadata and experimental conditions
- Quality control metrics
- Processing logs and summaries
- Statistical analyses and aggregations

## Development

### Setup Development Environment

```bash
# Install development dependencies
uv sync --extra dev

# Set up pre-commit hooks
pre-commit install
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test categories
uv run pytest -m "not slow"  # Skip slow tests
uv run pytest -m integration  # Only integration tests
```

### Code Quality

```bash
# Format code
uv run black src/

# Lint code
uv run ruff check src/

# Type checking
uv run mypy src/
```

## Dependencies

### Core Dependencies
- **mne** >= 1.6.0: EEG data processing
- **mne-bids** >= 0.15.0: BIDS format support
- **polars** >= 0.20.0: Efficient data processing
- **numpy** >= 1.25.0: Numerical computing
- **pandas** >= 2.1.0: Data manipulation (MNE compatibility)
- **scipy** >= 1.11.0: Scientific computing

### Machine Learning & Artifacts
- **scikit-learn** >= 1.3.0: Machine learning utilities
- **autoreject** >= 0.4.0: Automatic artifact rejection

### Visualization
- **matplotlib** >= 3.8.0: Plotting
- **seaborn** >= 0.13.0: Statistical visualization

### Development Tools
- **black** >= 23.0.0: Code formatting
- **ruff** >= 0.1.0: Linting and formatting
- **mypy** >= 1.7.0: Type checking
- **pytest** >= 7.4.0: Testing framework

## Output Files

The pipeline generates the following output files for each subject:

```
output_dir/
└── sub-01/
    └── eeg/
        ├── sub-01_preprocessed_raw.fif      # Preprocessed continuous data
        ├── sub-01_ica.fif                   # ICA components
        ├── sub-01_epoched.fif               # Final epochs
        ├── sub-01_metadata.csv              # Event metadata
        ├── checkpoints/                     # Intermediate results
        ├── figures/                         # Quality control plots
        └── logs/
            └── sub-01_processing_log.json   # Processing log
```

## Quality Control

The pipeline includes comprehensive quality control:

- **Channel Quality**: Automatic detection of flat, noisy, and correlated channels
- **Artifact Detection**: ASR-like burst rejection and ICA component analysis
- **Data Validation**: BIDS structure validation and file integrity checks
- **Processing Logs**: Detailed logging of all processing stages
- **Checkpoint System**: Save/load intermediate results for debugging

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `n_components` for ICA or process in smaller batches
2. **Channel Detection**: Adjust thresholds in configuration file
3. **Event Coding**: Verify trigger codes match your experimental setup
4. **BIDS Validation**: Ensure data follows BIDS format requirements

### Debug Mode

Enable verbose logging:
```bash
uv run eeg-pipeline --bids-root /path/to/bids --output /path/to/output --subject sub-01 --verbose
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this pipeline in your research, please cite:

```
Prior RDM Analysis: Python Implementation of EEG Preprocessing for Adaptive Control Experiments
Julius Kricheldorff
https://github.com/shichik07/Prior_RDM_Analysis
```

## Contact

- **Julius Kricheldorff**: julius@kricheldorff.de
- **Project Homepage**: https://julius-kricheldorff.com/
- **Repository**: https://github.com/shichik07/Prior_RDM_Analysis

## Acknowledgments

- Original MATLAB implementation by Julius Kricheldorff and Julia Ficke
- MNE-Python team for the excellent neuroimaging toolkit
- Polars team for efficient data processing framework
- EEGlab community for the original preprocessing methods
