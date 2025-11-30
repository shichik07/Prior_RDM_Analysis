"""
EEG Data Loading Module

Handles loading of EEG data from BIDS format using MNE-BIDS.
Provides validation and standardized Raw object creation.
"""

import mne
from mne_bids import BIDSPath, read_raw_bids
import os
from pathlib import Path
from typing import Optional, Dict, Any


def load_eeg_data(subject_id: str, bids_root: str, task: str = 'adaptivecontrol') -> mne.io.Raw:
    """
    Load EEG data from BIDS format
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier (e.g., 'sub-01')
    bids_root : str
        Path to BIDS root directory
    task : str
        Task name (default: 'adaptivecontrol')
    
    Returns:
    --------
    raw : mne.io.Raw
        Raw EEG data
    """
    bids_path = BIDSPath(
        subject=subject_id, 
        task=task, 
        suffix='eeg', 
        extension='.vhdr', 
        root=bids_root
    )
    
    try:
        raw = read_raw_bids(bids_path, verbose=False)
        return raw
    except Exception as e:
        raise FileNotFoundError(f"Could not load data for {subject_id}: {e}")


def setup_bids_paths(subject_id: str, bids_root: str, task: str = 'adaptivecontrol') -> Dict[str, str]:
    """
    Setup BIDS paths for a subject
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    bids_root : str
        Path to BIDS root directory
    task : str
        Task name
    
    Returns:
    --------
    paths : dict
        Dictionary containing relevant BIDS paths
    """
    paths = {
        'bids_root': bids_root,
        'subject_dir': os.path.join(bids_root, subject_id),
        'eeg_dir': os.path.join(bids_root, subject_id, 'eeg'),
        'derivatives_dir': os.path.join(bids_root, 'derivatives', subject_id, 'eeg')
    }
    
    return paths


def validate_bids_structure(bids_root: str) -> bool:
    """
    Validate BIDS directory structure
    
    Parameters:
    -----------
    bids_root : str
        Path to BIDS root directory
    
    Returns:
    --------
    valid : bool
        True if structure is valid BIDS
    """
    required_files = ['dataset_description.json']
    required_dirs = ['sub-01']
    
    bids_path = Path(bids_root)
    
    # Check required files
    for file in required_files:
        if not (bids_path / file).exists():
            print(f"Missing required BIDS file: {file}")
            return False
    
    # Check for at least one subject directory
    subject_dirs = [d for d in bids_path.iterdir() if d.is_dir() and d.name.startswith('sub-')]
    if not subject_dirs:
        print("No subject directories found (should start with 'sub-')")
        return False
    
    return True


def get_subject_list(bids_root: str) -> list:
    """
    Get list of all subjects in BIDS dataset
    
    Parameters:
    -----------
    bids_root : str
        Path to BIDS root directory
    
    Returns:
    --------
    subjects : list
        List of subject IDs
    """
    bids_path = Path(bids_root)
    subject_dirs = [d.name for d in bids_path.iterdir() 
                   if d.is_dir() and d.name.startswith('sub-')]
    
    return sorted(subject_dirs)


def check_eeg_files_exist(subject_id: str, bids_root: str, task: str = 'adaptivecontrol') -> bool:
    """
    Check if EEG files exist for a subject
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    bids_root : str
        Path to BIDS root directory
    task : str
        Task name
    
    Returns:
    --------
    exists : bool
        True if files exist
    """
    eeg_dir = os.path.join(bids_root, subject_id, 'eeg')
    if not os.path.exists(eeg_dir):
        return False
    
    # Check for BrainVision files
    vhdr_files = [f for f in os.listdir(eeg_dir) if f.endswith('.vhdr')]
    return len(vhdr_files) > 0
