"""
EEG Epoching and Final Processing Module

Completes the pipeline with ICA application, channel interpolation,
epoch extraction, and checkpoint system implementation.
"""

import mne
import numpy as np
import pandas as pd
import os
from datetime import datetime
from typing import Tuple, Dict, List, Optional
from .events import create_event_dictionary
from .utils import create_output_structure, log_processing_stage


def final_processing(raw: mne.io.Raw, ica: mne.preprocessing.ICA, 
                    metadata: pd.DataFrame, subject_id: str, 
                    output_dir: str) -> mne.Epochs:
    """
    Complete final processing chain
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    ica : mne.preprocessing.ICA
        Fitted ICA object
    metadata : pd.DataFrame
        Event metadata
    subject_id : str
        Subject identifier
    output_dir : str
        Output directory path
    
    Returns:
    --------
    epochs : mne.Epochs
        Final epoched data
    """
    # 1. Apply ICA and interpolate bad channels
    raw = apply_ica_and_interpolate(raw, ica)
    
    # 2. Create event dictionary
    event_dict = create_event_dictionary()
    
    # 3. Extract epochs
    epochs = extract_epochs(raw, metadata, event_dict)
    
    # 4. Save checkpoints
    save_checkpoints(raw, epochs, subject_id, output_dir, ica)
    
    return epochs


def apply_ica_and_interpolate(raw: mne.io.Raw, ica: mne.preprocessing.ICA) -> mne.io.Raw:
    """
    Apply ICA and interpolate bad channels
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    ica : mne.preprocessing.ICA
        Fitted ICA object
    
    Returns:
    --------
    raw : mne.io.Raw
        Raw data with ICA applied and channels interpolated
    """
    # Apply ICA to remove bad components
    if ica.exclude:
        raw = ica.apply(raw)
        print(f"Applied ICA, removed components: {ica.exclude}")
    else:
        print("No ICA components to remove")
    
    # Interpolate bad channels
    if raw.info['bads']:
        raw = raw.interpolate_bads()
        print(f"Interpolated bad channels: {raw.info['bads']}")
        # Clear bads list after interpolation
        raw.info['bads'] = []
    else:
        print("No bad channels to interpolate")
    
    return raw


def extract_epochs(raw: mne.io.Raw, metadata: pd.DataFrame, 
                  event_dict: Dict[str, int]) -> mne.Epochs:
    """
    Extract epochs around stimulus onset events
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    metadata : pd.DataFrame
        Event metadata
    event_dict : dict
        Event dictionary for epoching
    
    Returns:
    --------
    epochs : mne.Epochs
        Extracted epochs
    """
    # Find events in raw data
    events = mne.find_events(raw, stim_channel='STI 014')
    
    # Filter events to only include relevant onset events
    # Based on MATLAB code: 'S 21' to 'S 84' (MI, MC, main_incon, main_con events)
    relevant_codes = list(range(21, 85))  # S21 to S84
    
    # Filter events
    relevant_events = events[np.isin(events[:, 2], relevant_codes)]
    
    # Create subset of event dictionary for relevant events
    relevant_event_dict = {k: v for k, v in event_dict.items() 
                          if v in relevant_codes}
    
    # Extract epochs with -0.2 to 1.0 second window (as in MATLAB)
    epochs = mne.Epochs(
        raw, 
        relevant_events, 
        event_id=relevant_event_dict,
        tmin=-0.2, 
        tmax=1.0, 
        baseline=None,
        metadata=metadata.iloc[:len(relevant_events)] if len(metadata) >= len(relevant_events) else None,
        preload=True,
        verbose=False
    )
    
    print(f"Extracted {len(epochs)} epochs")
    
    return epochs


def save_checkpoints(raw: mne.io.Raw, epochs: mne.Epochs, 
                    subject_id: str, output_dir: str, 
                    ica: Optional[mne.preprocessing.ICA] = None) -> None:
    """
    Save intermediate results for debugging and quality control
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    epochs : mne.Epochs
        Epoched data
    subject_id : str
        Subject identifier
    output_dir : str
        Output directory path
    ica : mne.preprocessing.ICA, optional
        ICA object to save
    """
    # Create output structure
    subject_dir = create_output_structure(output_dir, subject_id)
    
    # Log processing stage
    log_processing_stage(subject_id, "epoching_completed", datetime.now())
    
    # Save preprocessed raw data
    raw_file = os.path.join(subject_dir, f'{subject_id}_preprocessed_raw.fif')
    raw.save(raw_file, overwrite=True)
    print(f"Saved preprocessed raw data: {raw_file}")
    
    # Save epochs
    epochs_file = os.path.join(subject_dir, f'{subject_id}_epoched.fif')
    epochs.save(epochs_file, overwrite=True)
    print(f"Saved epochs: {epochs_file}")
    
    # Save ICA components if available
    if ica is not None:
        ica_file = os.path.join(subject_dir, f'{subject_id}_ica.fif')
        ica.save(ica_file)
        print(f"Saved ICA: {ica_file}")
    
    # Save event metadata
    if hasattr(epochs, 'metadata') and epochs.metadata is not None:
        metadata_file = os.path.join(subject_dir, f'{subject_id}_metadata.csv')
        epochs.metadata.to_csv(metadata_file, index=False)
        print(f"Saved event metadata: {metadata_file}")


def create_epochs_summary(epochs: mne.Epochs) -> pd.DataFrame:
    """
    Create summary statistics for epochs
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epoched data
    
    Returns:
    --------
    summary : pd.DataFrame
        Summary statistics
    """
    summary_data = []
    
    for event_type in epochs.event_id.keys():
        event_epochs = epochs[event_type]
        
        summary_data.append({
            'event_type': event_type,
            'n_epochs': len(event_epochs),
            'mean_amplitude': np.mean(np.abs(event_epochs.get_data())),
            'std_amplitude': np.std(np.abs(event_epochs.get_data())),
            'duration': event_epochs.tmax - event_epochs.tmin
        })
    
    summary = pd.DataFrame(summary_data)
    return summary


def apply_baseline_correction(epochs: mne.Epochs, 
                            baseline: Tuple[float, float] = (-0.2, 0.0)) -> mne.Epochs:
    """
    Apply baseline correction to epochs
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epoched data
    baseline : tuple
        Baseline interval (start, end) in seconds
    
    Returns:
    --------
    epochs : mne.Epochs
        Baseline-corrected epochs
    """
    epochs.apply_baseline(baseline=baseline)
    return epochs


def reject_bad_epochs(epochs: mne.Epochs, 
                     reject_criteria: Optional[Dict[str, float]] = None) -> mne.Epochs:
    """
    Reject bad epochs based on amplitude criteria
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epoched data
    reject_criteria : dict, optional
        Rejection criteria for channels
    
    Returns:
    --------
    epochs : mne.Epochs
        Cleaned epochs
    """
    if reject_criteria is None:
        # Default rejection criteria
        reject_criteria = {
            'eeg': 100e-6,  # 100 microvolts
            'eog': 200e-6   # 200 microvolts for EOG
        }
    
    # Drop bad epochs
    epochs.drop_bad(reject=reject_criteria, verbose=False)
    
    print(f"Rejected {len(epochs.drop_log)} bad epochs")
    
    return epochs


def create_evoked_responses(epochs: mne.Epochs) -> Dict[str, mne.Evoked]:
    """
    Create evoked responses for each condition
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epoched data
    
    Returns:
    --------
    evoked_dict : dict
        Dictionary of evoked responses
    """
    evoked_dict = {}
    
    for condition in epochs.event_id.keys():
        evoked = epochs[condition].average()
        evoked_dict[condition] = evoked
    
    return evoked_dict


def save_evoked_responses(evoked_dict: Dict[str, mne.Evoked], 
                         subject_id: str, output_dir: str) -> None:
    """
    Save evoked responses
    
    Parameters:
    -----------
    evoked_dict : dict
        Dictionary of evoked responses
    subject_id : str
        Subject identifier
    output_dir : str
        Output directory path
    """
    subject_dir = create_output_structure(output_dir, subject_id)
    
    for condition, evoked in evoked_dict.items():
        evoked_file = os.path.join(subject_dir, f'{subject_id}_{condition}_evoked.fif')
        evoked.save(evoked_file)
        print(f"Saved evoked response: {evoked_file}")


def quality_control_metrics(epochs: mne.Epochs) -> Dict[str, float]:
    """
    Compute quality control metrics for epochs
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epoched data
    
    Returns:
    --------
    metrics : dict
        Quality control metrics
    """
    # Compute various quality metrics
    data = epochs.get_data()
    
    metrics = {
        'n_epochs': len(epochs),
        'n_channels': data.shape[1],
        'n_times': data.shape[2],
        'mean_amplitude': np.mean(np.abs(data)),
        'std_amplitude': np.std(data),
        'max_amplitude': np.max(np.abs(data)),
        'epoch_rejection_rate': len(epochs.drop_log) / (len(epochs) + len(epochs.drop_log)) if epochs.drop_log else 0.0
    }
    
    # Compute signal-to-noise ratio (simplified)
    if len(epochs) > 1:
        signal = np.mean(data, axis=0)  # Mean across epochs
        noise = np.std(data, axis=0)    # Std across epochs
        snr = np.mean(signal) / np.mean(noise)
        metrics['snr'] = snr
    
    return metrics
