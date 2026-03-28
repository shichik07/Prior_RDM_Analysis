"""
EEG Preprocessing Pipeline Module

Implements the core preprocessing steps including channel configuration,
resampling, temporal data rejection, filtering, and referencing.
"""

import mne
import numpy as np
import pandas as pd
from typing import Tuple, List
from .events import get_block_events


def preprocess_pipeline(raw: mne.io.Raw, metadata: pd.DataFrame) -> mne.io.Raw:
    """
    Main preprocessing chain
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    metadata : pd.DataFrame
        Event metadata
    
    Returns:
    --------
    raw : mne.io.Raw
        Preprocessed raw data
    """
    # 1. Channel configuration
    raw = configure_channels(raw)
    
    # 2. Add FCz channel
    raw = add_fcz_channel(raw)
    
    # 3. Set montage for channel locations
    raw = setup_montage(raw)
    
    # 4. Resample to 250 Hz
    raw.resample(250, npad="auto")
    
    # 5. Temporal rejection (outside experimental blocks)
    raw = reject_experimental_blocks(raw, metadata)
    
    # 6. Filtering
    raw = apply_filtering(raw)
    
    # 7. Referencing
    raw = setup_referencing(raw)
    
    return raw


def configure_channels(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Configure channel types and remove EOG channels
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    
    Returns:
    --------
    raw : mne.io.Raw
        Raw data with configured channels
    """
    # Set channel types for EOG channels
    eog_channels = ['HEOG', 'VEOG']
    existing_eog = [ch for ch in eog_channels if ch in raw.ch_names]
    
    if existing_eog:
        raw.set_channel_types({ch: 'eog' for ch in existing_eog})
    
    return raw


def add_fcz_channel(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Add FCz channel with proper coordinates
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    
    Returns:
    --------
    raw : mne.io.Raw
        Raw data with FCz channel added
    """
    if 'FCz' not in raw.ch_names:
        # Create FCz channel info with proper coordinates
        fcz_info = mne.create_info(
            ['FCz'], 
            raw.info['sfreq'], 
            ch_types='eeg'
        )
        
        # Set channel coordinates (from MATLAB code)
        fcz_info['chs'][0]['loc'] = np.array([
            0.390731128489274,  # X
            0.0,                # Y
            0.920504853452440,  # Z
            0.0,                # X for sphere
            0.0,                # Y for sphere
            1.0,                # Z for sphere
            0.127777777777778,  # radius
            0.0,                # theta
            67.0 * np.pi / 180  # phi in radians
        ])
        
        # Create zero-filled FCz data
        fcz_data = np.zeros((1, len(raw.times)))
        fcz_raw = mne.io.RawArray(fcz_data, fcz_info)
        
        # Add FCz channel to raw data
        raw.add_channels([fcz_raw])
    
    return raw


def setup_montage(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Setup channel montage for proper electrode locations
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    
    Returns:
    --------
    raw : mne.io.Raw
        Raw data with montage set
    """
    # Use standard 10-20 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False)
    
    return raw


def reject_experimental_blocks(raw: mne.io.Raw, metadata: pd.DataFrame) -> mne.io.Raw:
    """
    Remove data outside experimental blocks using annotations
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    metadata : pd.DataFrame
        Event metadata
    
    Returns:
    --------
    raw : mne.io.Raw
        Raw data with only experimental blocks
    """
    # Get block events (now returns integers)
    block_events = get_block_events()
    
    # Find block onset and offset events
    events = mne.find_events(raw)
    
    # Create annotations for bad segments (outside experimental blocks)
    bad_annotations = []
    
    # Add annotation for beginning of recording (before first block)
    first_onset_time = None
    for onset_code in block_events['onset']:
        onset_indices = np.where(events[:, 2] == onset_code)[0]
        if len(onset_indices) > 0:
            first_onset_sample = events[onset_indices[0], 0]
            first_onset_time = first_onset_sample / raw.info['sfreq'] - 1.5
            break
    
    if first_onset_time is not None and first_onset_time > 0:
        bad_annotations.append({
            'onset': 0,
            'duration': first_onset_time,
            'description': 'BAD_pre_experiment'
        })
    
    # Add annotations between blocks and after last block
    block_times = []
    for onset_code in block_events['onset']:
        for offset_code in block_events['offset']:
            onset_indices = np.where(events[:, 2] == onset_code)[0]
            offset_indices = np.where(events[:, 2] == offset_code)[0]
            
            for onset_idx in onset_indices:
                for offset_idx in offset_indices:
                    if onset_idx < offset_idx:
                        onset_time = events[onset_idx, 0] / raw.info['sfreq'] - 1.5
                        offset_time = events[offset_idx, 0] / raw.info['sfreq'] + 1.5
                        block_times.append((max(0, onset_time), offset_time))
    
    # Sort blocks by time
    block_times.sort()
    
    # Add bad segments between blocks
    for i in range(len(block_times) - 1):
        gap_start = block_times[i][1]
        gap_end = block_times[i + 1][0]
        if gap_end > gap_start:
            bad_annotations.append({
                'onset': gap_start,
                'duration': gap_end - gap_start,
                'description': 'BAD_between_blocks'
            })
    
    # Add annotation for end of recording (after last block)
    if block_times:
        last_block_end = block_times[-1][1]
        recording_duration = raw.times[-1]
        if recording_duration > last_block_end:
            bad_annotations.append({
                'onset': last_block_end,
                'duration': recording_duration - last_block_end,
                'description': 'BAD_post_experiment'
            })
    
    # Create annotations object
    if bad_annotations:
        annotations = mne.Annotations(
            onset=[a['onset'] for a in bad_annotations],
            duration=[a['duration'] for a in bad_annotations],
            description=[a['description'] for a in bad_annotations]
        )
        raw.set_annotations(raw.annotations + annotations)
    
    return raw


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merge overlapping intervals
    
    Parameters:
    -----------
    intervals : list of tuples
        List of (start, end) intervals
    
    Returns:
    --------
    merged : list of tuples
        Merged intervals
    """
    if not intervals:
        return []
    
    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        
        # Check if intervals overlap
        if current[0] <= last[1]:
            # Merge intervals
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    
    return merged


def apply_filtering(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Apply highpass filter and detrending
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    
    Returns:
    --------
    raw : mne.io.Raw
        Filtered raw data
    """
    # Highpass filter at 0.1 Hz
    raw.filter(
        l_freq=0.1, 
        h_freq=None, 
        method='fir', 
        fir_design='firwin',
        phase='zero'
    )
    
    # Detrend data
    raw._data = mne.preprocessing.detrend(raw._data)
    
    return raw


def setup_referencing(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Setup FCz average referencing
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    
    Returns:
    --------
    raw : mne.io.Raw
        Re-referenced raw data
    """
    # Set FCz as reference
    if 'FCz' in raw.ch_names:
        raw.set_eeg_reference(ref_channels=['FCz'], projection=False)
    else:
        # If FCz not available, use average reference
        raw.set_eeg_reference(ref_channels='average', projection=False)
    
    return raw


def remove_eog_channels(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Remove EOG channels from data
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    
    Returns:
    --------
    raw : mne.io.Raw
        Raw data without EOG channels
    """
    eog_channels = ['HEOG', 'VEOG']
    existing_eog = [ch for ch in eog_channels if ch in raw.ch_names]
    
    if existing_eog:
        raw.drop_channels(existing_eog)
    
    return raw
