"""
EEG Event Processing Module

Handles the random-dot motion task event coding system.
Translates trigger codes to experimental condition metadata including
info conditions (mono, null, part, full), coherence levels, and response types.
"""

import mne
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List


def code_events(raw: mne.io.Raw) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Translate MATLAB event coding to Python
    
    Implements the complex SXXX coding logic from the original MATLAB code
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    
    Returns:
    --------
    events : np.ndarray
        MNE events array
    metadata : pd.DataFrame
        Event metadata with experimental condition information
    """
    events = mne.find_events(raw)
    
    # Handle incorrect response event renaming first
    modified_events = handle_incorrect_responses(events, raw)
    
    # Create metadata DataFrame using modified events
    metadata = create_event_metadata(modified_events)
    
    return modified_events, metadata


def handle_incorrect_responses(events: np.ndarray, raw: mne.io.Raw) -> np.ndarray:
    """
    Handle response events for random-dot motion task
    
    In the random-dot motion task, response codes are independent events
    and don't modify preceding onset events like the stroop task.
    This function returns events unchanged as the new task structure
    treats responses as separate events.
    
    Parameters:
    -----------
    events : np.ndarray
        MNE events array
    raw : mne.io.Raw
        Raw EEG data
    
    Returns:
    --------
    events : np.ndarray
        Unmodified events array (responses are independent events)
    """
    # For random-dot motion task, response codes are independent
    # and don't modify onset events like in the stroop task
    return events.copy()


def parse_sxxx_code(event_code: int) -> Dict[str, str]:
    """
    Parse random-dot motion task trigger code into experimental conditions
    
    Parameters:
    -----------
    event_code : int
        Event trigger code (integer from MNE events)
    
    Returns:
    --------
    conditions : dict
        Dictionary with condition information
    """
    conditions = {
        'part': '',
        'info_condition': '',
        'coherence': '',
        'response_type': '',
        'response_accuracy': ''
    }
    
    # Handle practice triggers
    if event_code == 200:
        conditions.update({
            'part': 'Practice Start',
            'info_condition': 'Practice Start',
            'coherence': 'Practice Start',
            'response_type': 'Practice Start',
            'response_accuracy': 'Practice Start'
        })
        return conditions
    elif event_code == 201:
        conditions.update({
            'part': 'Practice End',
            'info_condition': 'Practice End',
            'coherence': 'Practice End',
            'response_type': 'Practice End',
            'response_accuracy': 'Practice End'
        })
        return conditions
    
    # Handle block triggers
    if 1 <= event_code <= 8:
        conditions.update({
            'part': 'Start Block',
            'info_condition': 'Start Block',
            'coherence': 'Start Block',
            'response_type': 'Start Block',
            'response_accuracy': 'Start Block'
        })
        return conditions
    elif 11 <= event_code <= 18:
        conditions.update({
            'part': 'End Block',
            'info_condition': 'End Block',
            'coherence': 'End Block',
            'response_type': 'End Block',
            'response_accuracy': 'End Block'
        })
        return conditions
    
    # Handle fixation
    if event_code == 20:
        conditions.update({
            'part': 'Fixation',
            'info_condition': 'Fixation',
            'coherence': 'Fixation',
            'response_type': 'Fixation',
            'response_accuracy': 'Fixation'
        })
        return conditions
    
    # Handle trial onsets (31-34, 41-44, 51-54, 61-64)
    if 31 <= event_code <= 64:
        conditions['part'] = 'Onset'
        
        # Extract info condition from tens digit
        tens_digit = event_code // 10
        if tens_digit == 3:
            conditions['info_condition'] = 'Mono_info'
        elif tens_digit == 4:
            conditions['info_condition'] = 'Null_info'
        elif tens_digit == 5:
            conditions['info_condition'] = 'Part_info'
        elif tens_digit == 6:
            conditions['info_condition'] = 'Full_info'
        
        # Extract coherence from units digit (1-4, low to high)
        units_digit = event_code % 10
        if units_digit == 1:
            conditions['coherence'] = 'low1'
        elif units_digit == 2:
            conditions['coherence'] = 'low2'
        elif units_digit == 3:
            conditions['coherence'] = 'high1'
        elif units_digit == 4:
            conditions['coherence'] = 'high2'
        
        conditions['response_type'] = 'Onset'
        conditions['response_accuracy'] = 'Onset'
        return conditions
    
    # Handle response codes
    if 130 <= event_code <= 169:
        conditions['part'] = 'Response'
        
        # Extract info condition from hundreds/tens digit
        if 130 <= event_code <= 139:
            conditions['info_condition'] = 'Mono_info'
        elif 140 <= event_code <= 149:
            conditions['info_condition'] = 'Null_info'
        elif 150 <= event_code <= 159:
            conditions['info_condition'] = 'Part_info'
        elif 160 <= event_code <= 169:
            conditions['info_condition'] = 'Full_info'
        
        # Determine response type and accuracy
        if event_code in [130, 140, 150, 160]:
            conditions['response_type'] = 'late'
            conditions['response_accuracy'] = 'late'
        elif event_code in [139, 149, 159, 169]:
            conditions['response_type'] = 'none'
            conditions['response_accuracy'] = 'none'
        elif event_code in [135, 136, 137, 138, 145, 146, 147, 148, 155, 156, 157, 158, 165, 166, 167, 168]:
            conditions['response_type'] = 'response'
            conditions['response_accuracy'] = 'false'
        elif event_code in [131, 132, 133, 134, 141, 142, 143, 144, 151, 152, 153, 154, 161, 162, 163, 164]:
            conditions['response_type'] = 'response'
            conditions['response_accuracy'] = 'correct'
        
        # Extract coherence from units digit for response codes
        # Only extract coherence for response codes that have coherence information
        if event_code in [131, 132, 133, 134, 135, 136, 137, 138, 
                          141, 142, 143, 144, 145, 146, 147, 148,
                          151, 152, 153, 154, 155, 156, 157, 158,
                          161, 162, 163, 164, 165, 166, 167, 168]:
            units_digit = event_code % 10
            if units_digit == 1:
                conditions['coherence'] = 'low1'
            elif units_digit == 2:
                conditions['coherence'] = 'low2'
            elif units_digit == 3:
                conditions['coherence'] = 'high1'
            elif units_digit == 4:
                conditions['coherence'] = 'high2'
            else:
                conditions['coherence'] = 'unknown'
        
        return conditions
    
    # Default case for unrecognized codes
    import warnings
    warnings.warn(f"Unknown event code encountered: {event_code}", UserWarning)
    conditions.update({
        'part': 'Unknown',
        'info_condition': 'Unknown',
        'coherence': 'Unknown',
        'response_type': 'Unknown',
        'response_accuracy': 'Unknown'
    })
    
    return conditions


def create_event_metadata(events: np.ndarray) -> pd.DataFrame:
    """
    Create metadata DataFrame from events
    
    Parameters:
    -----------
    events : np.ndarray
        MNE events array
    
    Returns:
    --------
    metadata : pd.DataFrame
        Event metadata with experimental conditions
    """
    metadata_list = []
    
    for event in events:
        event_code = event[2]
        conditions = parse_sxxx_code(event_code)
        metadata_list.append(conditions)
    
    metadata = pd.DataFrame(metadata_list)
    return metadata


def create_event_dictionary() -> Dict[str, int]:
    """
    Create event dictionary for epoching - random-dot motion task
    
    Returns:
    --------
    event_dict : dict
        Mapping of event names to trigger codes
    """
    event_dict = {
        # Practice events
        'practice_start': 200,
        'practice_end': 201,
        
        # Block events
        'start_block_1': 1, 'start_block_2': 2, 'start_block_3': 3, 'start_block_4': 4,
        'start_block_5': 5, 'start_block_6': 6, 'start_block_7': 7, 'start_block_8': 8,
        'end_block_1': 11, 'end_block_2': 12, 'end_block_3': 13, 'end_block_4': 14,
        'end_block_5': 15, 'end_block_6': 16, 'end_block_7': 17, 'end_block_8': 18,
        
        # Fixation
        'fixation': 20,
        
        # Trial onsets by info condition and coherence
        'mono_info_low1': 31, 'mono_info_low2': 32, 'mono_info_high1': 33, 'mono_info_high2': 34,
        'null_info_low1': 41, 'null_info_low2': 42, 'null_info_high1': 43, 'null_info_high2': 44,
        'part_info_low1': 51, 'part_info_low2': 52, 'part_info_high1': 53, 'part_info_high2': 54,
        'full_info_low1': 61, 'full_info_low2': 62, 'full_info_high1': 63, 'full_info_high2': 64,
        
        # Correct responses by info condition and coherence
        'mono_info_correct_low1': 131, 'mono_info_correct_low2': 132, 'mono_info_correct_high1': 133, 'mono_info_correct_high2': 134,
        'null_info_correct_low1': 141, 'null_info_correct_low2': 142, 'null_info_correct_high1': 143, 'null_info_correct_high2': 144,
        'part_info_correct_low1': 151, 'part_info_correct_low2': 152, 'part_info_correct_high1': 153, 'part_info_correct_high2': 154,
        'full_info_correct_low1': 161, 'full_info_correct_low2': 162, 'full_info_correct_high1': 163, 'full_info_correct_high2': 164,
        
        # False responses by info condition and coherence
        'mono_info_false_low1': 135, 'mono_info_false_low2': 136, 'mono_info_false_high1': 137, 'mono_info_false_high2': 138,
        'null_info_false_low1': 145, 'null_info_false_low2': 146, 'null_info_false_high1': 147, 'null_info_false_high2': 148,
        'part_info_false_low1': 155, 'part_info_false_low2': 156, 'part_info_false_high1': 157, 'part_info_false_high2': 158,
        'full_info_false_low1': 165, 'full_info_false_low2': 166, 'full_info_false_high1': 167, 'full_info_false_high2': 168,
        
        # No response by info condition
        'mono_info_none': 139,
        'null_info_none': 149,
        'part_info_none': 159,
        'full_info_none': 169,
        
        # Late response by info condition
        'mono_info_late': 130,
        'null_info_late': 140,
        'part_info_late': 150,
        'full_info_late': 160,
    }
    
    return event_dict


def get_block_events() -> Dict[str, List[int]]:
    """
    Get block onset and offset trigger codes for random-dot motion task
    
    Returns:
    --------
    block_events : dict
        Dictionary with block onset and offset trigger codes as integers
    """
    block_events = {
        'onset': [1, 2, 3, 4, 5, 6, 7, 8],      # Start Block 1-8
        'offset': [11, 12, 13, 14, 15, 16, 17, 18],  # End Block 1-8
        'practice_start': 200,  # Practice Start
        'practice_end': 201     # Practice End
    }
    
    return block_events
