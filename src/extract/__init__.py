"""
EEG Preprocessing Extraction Package

This package provides a complete Python implementation of the MATLAB EEG preprocessing
pipeline for adaptive control experiments, using MNE-Python.

Modules:
--------
loading: BIDS data loading and validation
events: Complex SXXX event coding system
preprocessing: Core preprocessing pipeline
artifacts: Advanced artifact removal and ICA
epoching: Final processing and epoch extraction
utils: Checkpoint system and utilities
"""

from .loading import load_eeg_data, setup_bids_paths, validate_bids_structure, get_subject_list
from .events import code_events, create_event_metadata, get_block_events
from .preprocessing import preprocess_pipeline, configure_channels, add_fcz_channel
from .artifacts import detect_bad_channels, asr_replacement, ica_pipeline, apply_line_noise_removal
from .epoching import final_processing, extract_epochs, save_checkpoints
from .utils import create_output_structure, save_checkpoint, load_checkpoint

__version__ = "1.0.0"
__author__ = "Julius Kricheldorff, Julia Ficke"

__all__ = [
    # Loading functions
    'load_eeg_data',
    'setup_bids_paths', 
    'validate_bids_structure',
    'get_subject_list',
    
    # Event functions
    'code_events',
    'create_event_metadata',
    'get_block_events',
    
    # Preprocessing functions
    'preprocess_pipeline',
    'configure_channels',
    'add_fcz_channel',
    
    # Artifact functions
    'detect_bad_channels',
    'asr_replacement',
    'ica_pipeline',
    'apply_line_noise_removal',
    
    # Epoching functions
    'final_processing',
    'extract_epochs',
    'save_checkpoints',
    
    # Utility functions
    'create_output_structure',
    'save_checkpoint',
    'load_checkpoint',
]
