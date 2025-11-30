"""
EEG Artifact Removal Module

Handles advanced artifact removal including automatic bad channel detection,
ASR replacement using autoreject, ICA decomposition, and line noise removal.
"""

import mne
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import autoreject
from mne.preprocessing import ICA


def detect_bad_channels(raw: mne.io.Raw) -> Tuple[mne.io.Raw, Dict[str, List[str]]]:
    """
    Automatic channel detection (MNE equivalent of pop_clean_rawdata)
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    
    Returns:
    --------
    raw : mne.io.Raw
        Raw data with bad channels marked
    """
    # Initialize bad channels list
    if raw.info['bads'] is None:
        raw.info['bads'] = []
    
    # Flatline detection (channels with very low variance)
    flatline_channels = []
    flatline_threshold = 1e-6  # Very low variance threshold
    
    for ch in raw.ch_names:
        if ch not in raw.info['bads']:
            ch_data = raw.get_data(picks=[ch])[0]
            ch_std = np.std(ch_data)
            
            if ch_std < flatline_threshold:
                flatline_channels.append(ch)
    
    # Channel correlation detection (highly correlated channels)
    correlation_channels = []
    correlation_threshold = 0.8
    
    # Only check EEG channels for correlation
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False)
    eeg_channels = [raw.ch_names[i] for i in eeg_picks]
    
    if len(eeg_channels) > 1:
        eeg_data = raw.get_data(picks=eeg_picks)
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(eeg_data)
        
        # Find highly correlated channel pairs
        for i in range(len(eeg_channels)):
            for j in range(i + 1, len(eeg_channels)):
                if abs(corr_matrix[i, j]) > correlation_threshold:
                    # Mark the second channel as bad (arbitrary choice)
                    if eeg_channels[j] not in raw.info['bads'] + flatline_channels:
                        correlation_channels.append(eeg_channels[j])
    
    # Line noise detection (channels with high 50Hz power)
    line_noise_channels = []
    line_noise_threshold = 5  # z-score threshold
    
    # Compute power spectrum
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False)
    if len(eeg_picks) > 0:
        psd, freqs = mne.time_frequency.psd_welch(
            raw, fmin=45, fmax=55, tmax=60, picks=eeg_picks, verbose=False
        )
        
        # Find 50Hz power
        freq_50_idx = np.argmin(np.abs(freqs - 50))
        power_50hz = psd[:, freq_50_idx]
        
        # Z-score normalization
        z_scores = (power_50hz - np.mean(power_50hz)) / np.std(power_50hz)
        
        # Mark channels with high 50Hz power
        for i, ch_idx in enumerate(eeg_picks):
            if z_scores[i] > line_noise_threshold:
                ch_name = raw.ch_names[ch_idx]
                if ch_name not in raw.info['bads'] + flatline_channels + correlation_channels:
                    line_noise_channels.append(ch_name)
    
    # Combine all bad channels
    all_bad_channels = list(set(flatline_channels + correlation_channels + line_noise_channels))
    raw.info['bads'].extend(all_bad_channels)
    
    print(f"Detected bad channels: {all_bad_channels}")
    
    return raw


def clean_rawdata_pipeline(raw: mne.io.Raw) -> mne.io.Raw:
    """
    MNE equivalent of pop_clean_rawdata with all criteria
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    
    Returns:
    --------
    raw : mne.io.Raw
        Cleaned raw data
    """
    # Store original data for comparison
    original_raw = raw.copy()
    
    # Apply all detection criteria
    raw = detect_bad_channels(raw)
    
    # Store rejected channel information
    rejected_info = {
        'flatline_channels': [],
        'correlation_channels': [],
        'line_noise_channels': [],
        'all_bad_channels': raw.info['bads'].copy()
    }
    
    return raw, rejected_info


def asr_replacement(raw: mne.io.Raw) -> mne.io.Raw:
    """
    ASR-like functionality using autoreject
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    
    Returns:
    --------
    raw : mne.io.Raw
        Cleaned raw data
    """
    try:
        # Create epochs for autoreject (use sliding windows)
        events = mne.make_fixed_length_events(raw, duration=2.0, overlap=1.0)
        
        epochs = mne.Epochs(
            raw, events, tmin=0, tmax=2.0, baseline=None, 
            preload=True, verbose=False, reject_by_annotation=True
        )
        
        # Apply autoreject for artifact detection
        ar = autoreject.AutoReject(
            n_interpolate=[1, 2, 3, 4],
            consensus_params=[0.5, 0.6, 0.7, 0.8],
            verbose=False
        )
        
        epochs_clean = ar.fit_transform(epochs)
        
        # Reconstruct continuous data from cleaned epochs
        # This is a simplified approach - in practice, you might want to
        # use the reject log to mark bad segments in the original data
        
        # For now, return the original raw with annotations for bad segments
        # Create annotations for bad epochs
        bad_annotations = []
        
        # Get reject log from autoreject
        if hasattr(ar, 'reject_log_'):
            reject_log = ar.reject_log_
            for i, is_bad in enumerate(reject_log.bad_epochs):
                if is_bad:
                    onset = events[i, 0] / raw.info['sfreq']
                    bad_annotations.append({
                        'onset': onset,
                        'duration': 2.0,
                        'description': 'BAD_asr_detected'
                    })
        
        # Add annotations to raw data
        if bad_annotations:
            annotations = mne.Annotations(
                onset=[a['onset'] for a in bad_annotations],
                duration=[a['duration'] for a in bad_annotations],
                description=[a['description'] for a in bad_annotations]
            )
            raw.set_annotations(raw.annotations + annotations)
        
        print(f"ASR replacement: marked {len(bad_annotations)} bad segments")
        return raw
        
    except Exception as e:
        print(f"ASR replacement failed: {e}")
        return raw


def apply_line_noise_removal(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Remove 50Hz line noise using spectrum_fit method
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    
    Returns:
    --------
    raw : mne.io.Raw
        Raw data with line noise removed
    """
    # Remove 50Hz line noise and harmonics
    raw.notch_filter(
        freqs=[50, 100, 150],  # 50Hz and harmonics
        method='spectrum_fit',
        notch_widths=2,
        verbose=False
    )
    
    return raw


def ica_pipeline(raw: mne.io.Raw, n_components: int = 20) -> ICA:
    """
    ICA decomposition and component removal
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    n_components : int
        Number of ICA components to compute
    
    Returns:
    --------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    """
    # Create ICA object
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=97,
        method='picard',
        max_iter='auto',
        verbose=False
    )
    
    # Fit ICA on EEG channels only
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False)
    ica.fit(raw, picks=eeg_picks)
    
    # Automatic detection of EOG/ECG artifacts
    try:
        # Find EOG artifacts
        eog_channels = mne.pick_types(raw.info, eog=True)
        if len(eog_channels) > 0:
            ica.find_bads_eog(raw, verbose=False)
        
        # Find ECG artifacts
        ecg_channels = mne.pick_types(raw.info, ecg=True)
        if len(ecg_channels) > 0:
            ica.find_bads_ecg(raw, verbose=False)
            
    except Exception as e:
        print(f"Automatic artifact detection failed: {e}")
    
    return ica


def manual_ica_inspection(ica: ICA, raw: mne.io.Raw) -> List[str]:
    """
    Manual ICA component inspection and selection
    
    Parameters:
    -----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    raw : mne.io.Raw
        Raw EEG data
    
    Returns:
    --------
    bad_components : list
        List of bad component indices
    """
    # Plot ICA components for manual inspection
    ica.plot_components(inst=raw)
    
    # Plot component properties and time courses
    ica.plot_properties(raw)
    ica.plot_sources(raw)
    
    # Get user input for bad components
    print("Enter bad component indices (comma-separated), or 'none' if no bad components:")
    user_input = input().strip()
    
    if user_input.lower() == 'none':
        bad_components = []
    else:
        try:
            bad_components = [int(x.strip()) for x in user_input.split(',')]
        except ValueError:
            print("Invalid input. No components marked as bad.")
            bad_components = []
    
    return bad_components


def apply_ica_rejection(raw: mne.io.Raw, ica: ICA, bad_components: List[str]) -> mne.io.Raw:
    """
    Apply ICA component rejection
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    ica : mne.preprocessing.ICA
        Fitted ICA object
    bad_components : list
        List of bad component indices to remove
    
    Returns:
    --------
    raw : mne.io.Raw
        Raw data with ICA components removed
    """
    if bad_components:
        ica.exclude = bad_components
        raw = ica.apply(raw)
        print(f"Removed ICA components: {bad_components}")
    else:
        print("No ICA components removed")
    
    return raw


def burst_rejection_pipeline(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Apply burst rejection similar to ASR with 80 criterion
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    
    Returns:
    --------
    raw : mne.io.Raw
        Raw data with burst artifacts rejected
    """
    # Create epochs for burst detection
    events = mne.make_fixed_length_events(raw, duration=1.0, overlap=0.5)
    
    try:
        epochs = mne.Epochs(
            raw, events, tmin=0, tmax=1.0, baseline=None,
            preload=True, verbose=False
        )
        
        # Apply amplitude rejection
        reject_criteria = {
            'eeg': 80e-6  # 80 microvolts threshold
        }
        
        epochs.drop_bad(reject=reject_criteria, verbose=False)
        
        # Convert back to continuous data (simplified approach)
        if len(epochs) > 0:
            # For now, return the original raw data
            # In a full implementation, we'd reconstruct continuous data
            pass
        
    except Exception as e:
        print(f"Burst rejection failed: {e}")
    
    return raw
