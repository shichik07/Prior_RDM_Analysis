"""
EEG Preprocessing Utilities Module

Provides utility functions for checkpoint system, file management,
logging, and general helper functions.
"""

import os
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import mne
import numpy as np
import pandas as pd


def create_output_structure(output_dir: str, subject_id: str) -> str:
    """
    Setup output directories for a subject
    
    Parameters:
    -----------
    output_dir : str
        Root output directory
    subject_id : str
        Subject identifier
    
    Returns:
    --------
    subject_dir : str
        Path to subject's output directory
    """
    subject_dir = os.path.join(output_dir, subject_id, 'eeg')
    
    # Create directory structure
    subdirs = [
        subject_dir,
        os.path.join(subject_dir, 'checkpoints'),
        os.path.join(subject_dir, 'figures'),
        os.path.join(subject_dir, 'logs')
    ]
    
    for subdir in subdirs:
        os.makedirs(subdir, exist_ok=True)
    
    return subject_dir


def save_checkpoint(data: Any, subject_id: str, stage: str, 
                   output_dir: str, format: str = 'fif') -> str:
    """
    Save intermediate data checkpoint
    
    Parameters:
    -----------
    data : Any
        Data to save (Raw, Epochs, ICA, etc.)
    subject_id : str
        Subject identifier
    stage : str
        Processing stage name
    output_dir : str
        Output directory
    format : str
        File format ('fif', 'pkl', 'json')
    
    Returns:
    --------
    filepath : str
        Path to saved file
    """
    checkpoint_dir = os.path.join(output_dir, subject_id, 'eeg', 'checkpoints')
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{subject_id}_{stage}_{timestamp}.{format}"
    filepath = os.path.join(checkpoint_dir, filename)
    
    # Save based on data type and format
    if isinstance(data, (mne.io.Raw, mne.Epochs)) and format == 'fif':
        data.save(filepath, overwrite=True)
    elif isinstance(data, mne.preprocessing.ICA) and format == 'fif':
        data.save(filepath)
    elif format == 'pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif format == 'json':
        if isinstance(data, pd.DataFrame):
            data.to_json(filepath, orient='records')
        else:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Saved checkpoint: {filepath}")
    return filepath


def load_checkpoint(subject_id: str, stage: str, output_dir: str, 
                   format: str = 'fif', timestamp: Optional[str] = None) -> Any:
    """
    Load saved checkpoint data
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    stage : str
        Processing stage name
    output_dir : str
        Output directory
    format : str
        File format ('fif', 'pkl', 'json')
    timestamp : str, optional
        Specific timestamp to load (if None, loads latest)
    
    Returns:
    --------
    data : Any
        Loaded data
    """
    checkpoint_dir = os.path.join(output_dir, subject_id, 'eeg', 'checkpoints')
    
    # Find checkpoint files
    pattern = f"{subject_id}_{stage}_*.{format}"
    checkpoint_files = list(Path(checkpoint_dir).glob(pattern))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint found for {subject_id}_{stage}")
    
    # Select file (latest or specific timestamp)
    if timestamp:
        target_file = Path(checkpoint_dir) / f"{subject_id}_{stage}_{timestamp}.{format}"
        if not target_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {target_file}")
    else:
        # Load latest file
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        target_file = checkpoint_files[0]
    
    # Load based on format
    if format == 'fif':
        if target_file.stem.endswith('_raw') or target_file.stem.endswith('_preprocessed'):
            return mne.io.read_raw_fif(str(target_file), preload=True)
        elif target_file.stem.endswith('_epoched'):
            return mne.read_epochs(str(target_file))
        elif target_file.stem.endswith('_ica'):
            return mne.preprocessing.read_ica(str(target_file))
    elif format == 'pkl':
        with open(target_file, 'rb') as f:
            return pickle.load(f)
    elif format == 'json':
        if target_file.suffix == '.json':
            return pd.read_json(str(target_file))
        else:
            with open(target_file, 'r') as f:
                return json.load(f)
    
    raise ValueError(f"Cannot load file: {target_file}")


def log_processing_stage(subject_id: str, stage: str, timestamp: datetime, 
                        output_dir: str, **kwargs) -> None:
    """
    Log processing stage information
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    stage : str
        Processing stage name
    timestamp : datetime
        Timestamp of stage completion
    output_dir : str
        Output directory
    **kwargs : dict
        Additional logging information
    """
    log_dir = os.path.join(output_dir, subject_id, 'eeg', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{subject_id}_processing_log.json")
    
    # Create log entry
    log_entry = {
        'subject_id': subject_id,
        'stage': stage,
        'timestamp': timestamp.isoformat(),
        **kwargs
    }
    
    # Load existing log or create new
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = {'processing_log': []}
    
    # Append new entry
    log_data['processing_log'].append(log_entry)
    
    # Save updated log
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)


def get_processing_log(subject_id: str, output_dir: str) -> Dict[str, Any]:
    """
    Get processing log for a subject
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    output_dir : str
        Output directory
    
    Returns:
    --------
    log_data : dict
        Processing log data
    """
    log_file = os.path.join(output_dir, subject_id, 'eeg', 'logs', f"{subject_id}_processing_log.json")
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            return json.load(f)
    else:
        return {'processing_log': []}


def create_processing_summary(subject_id: str, output_dir: str) -> pd.DataFrame:
    """
    Create summary of processing stages
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    output_dir : str
        Output directory
    
    Returns:
    --------
    summary : pd.DataFrame
        Processing summary
    """
    log_data = get_processing_log(subject_id, output_dir)
    
    if not log_data['processing_log']:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(log_data['processing_log'])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate duration between stages
    df = df.sort_values('timestamp')
    df['duration_seconds'] = df['timestamp'].diff().dt.total_seconds()
    
    return df


def backup_file(filepath: str, backup_dir: str) -> str:
    """
    Create backup of a file
    
    Parameters:
    -----------
    filepath : str
        Path to file to backup
    backup_dir : str
        Backup directory
    
    Returns:
    --------
    backup_path : str
        Path to backup file
    """
    os.makedirs(backup_dir, exist_ok=True)
    
    filename = os.path.basename(filepath)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{timestamp}_{filename}"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    if os.path.exists(filepath):
        import shutil
        shutil.copy2(filepath, backup_path)
        print(f"Created backup: {backup_path}")
    
    return backup_path


def validate_file_integrity(filepath: str, expected_size: Optional[int] = None) -> bool:
    """
    Validate file integrity
    
    Parameters:
    -----------
    filepath : str
        Path to file
    expected_size : int, optional
        Expected file size in bytes
    
    Returns:
    --------
    valid : bool
        True if file is valid
    """
    if not os.path.exists(filepath):
        return False
    
    # Check file size
    file_size = os.path.getsize(filepath)
    if expected_size is not None and file_size != expected_size:
        return False
    
    # Try to read file based on extension
    try:
        if filepath.endswith('.fif'):
            if 'raw' in filepath:
                mne.io.read_raw_fif(filepath, preload=False)
            elif 'epoched' in filepath:
                mne.read_epochs(filepath)
        elif filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                pickle.load(f)
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                json.load(f)
        
        return True
    except Exception as e:
        print(f"File validation failed: {e}")
        return False


def cleanup_old_checkpoints(subject_id: str, output_dir: str, 
                           keep_latest: int = 3) -> None:
    """
    Clean up old checkpoint files, keeping only the latest ones
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    output_dir : str
        Output directory
    keep_latest : int
        Number of latest files to keep per stage
    """
    checkpoint_dir = os.path.join(output_dir, subject_id, 'eeg', 'checkpoints')
    
    if not os.path.exists(checkpoint_dir):
        return
    
    # Group files by stage
    stage_files = {}
    for file in Path(checkpoint_dir).glob(f"{subject_id}_*"):
        parts = file.stem.split('_')
        if len(parts) >= 3:
            stage = '_'.join(parts[1:-1])  # Extract stage name
            if stage not in stage_files:
                stage_files[stage] = []
            stage_files[stage].append(file)
    
    # Keep only latest files per stage
    for stage, files in stage_files.items():
        # Sort by modification time
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old files
        for old_file in files[keep_latest:]:
            old_file.unlink()
            print(f"Removed old checkpoint: {old_file}")


def create_config_file(output_dir: str, config: Dict[str, Any]) -> str:
    """
    Create configuration file
    
    Parameters:
    -----------
    output_dir : str
        Output directory
    config : dict
        Configuration parameters
    
    Returns:
    --------
    config_file : str
        Path to config file
    """
    config_file = os.path.join(output_dir, 'preprocessing_config.json')
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created config file: {config_file}")
    return config_file


def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration file
    
    Parameters:
    -----------
    config_path : str
        Path to config file
    
    Returns:
    --------
    config : dict
        Configuration parameters
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config
