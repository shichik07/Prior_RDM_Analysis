"""
EEG Event Processing Module

Handles the complex SXXX event coding system from the MATLAB pipeline.
Translates trigger codes to experimental condition metadata.
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
    Rename preceding onset events based on incorrect response codes
    
    Implements the logic from MATLAB lines 65-103 where S131-S194 trigger
    codes cause the preceding onset event to be renamed.
    
    Parameters:
    -----------
    events : np.ndarray
        MNE events array
    raw : mne.io.Raw
        Raw EEG data
    
    Returns:
    --------
    events : np.ndarray
        Modified events array
    """
    # Create a copy to modify
    modified_events = events.copy()
    
    # Mapping of response codes to new onset codes
    response_mapping = {
        131: 31, 132: 32, 133: 33, 134: 34,  # S131-S134 -> S31-S34
        151: 51, 152: 52, 153: 53, 154: 54,  # S151-S154 -> S51-S54
        171: 71, 172: 72, 173: 73, 174: 74,  # S171-S174 -> S71-S74
        191: 91, 192: 92, 193: 93, 194: 94   # S191-S194 -> S91-S94
    }
    
    # Iterate through events (excluding the last one)
    for i in range(len(events) - 1):
        current_event_code = events[i, 2]
        next_event_code = events[i + 1, 2]
        
        # Check if next event is an incorrect response
        if next_event_code in response_mapping:
            # Rename current onset event
            modified_events[i, 2] = response_mapping[next_event_code]
    
    return modified_events


def parse_sxxx_code(event_code: int) -> Dict[str, str]:
    """
    Parse SXXX trigger code into experimental conditions
    
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
        'analysetype': '',
        'congruency': '',
        'trial': '',
        'answer': ''
    }
    
    # Handle non-stimulus codes (convert integer to string format for parsing)
    code_str = str(event_code)
    
    # Handle start/end blocks and other non-experimental codes
    if event_code < 20 or event_code > 200:
        conditions.update({
            'part': 'Start Block',
            'analysetype': 'Start Block',
            'congruency': 'Start Block',
            'trial': 'Start Block',
            'answer': 'Start Block'
        })
        return conditions
    
    # Parse experimental codes (21-94 range)
    try:
        # Extract digits from integer
        if event_code >= 21 and event_code <= 94:
            digit1 = event_code // 10  # Tens digit
            digit2 = event_code % 10   # Units digit
            
            # For three-digit codes (like 131-194), handle differently
            if event_code >= 100:
                digit1 = (event_code // 100) % 10  # Hundreds digit
                digit2 = (event_code // 10) % 10   # Tens digit  
                digit3 = event_code % 10           # Units digit
            else:
                digit3 = digit2  # For two-digit codes, units digit determines condition
                digit2 = digit1  # Tens digit determines analysetype
        else:
            return conditions
        
        # Part determination (onset, response, fixation)
        if digit1 == 0:
            conditions['part'] = 'Onset'
            conditions['answer'] = 'Onset'
        elif digit1 == 1:
            conditions['part'] = 'Response'
        elif digit1 == 2:
            conditions['part'] = 'Fixation'
        
        # Handle start/end blocks for special cases
        if event_code in [1, 101, 102, 103, 104, 105, 106, 107, 108]:
            if event_code == 1:
                conditions.update({
                    'part': 'Start Block',
                    'analysetype': 'Start Block',
                    'congruency': 'Start Block',
                    'trial': 'Start Block',
                    'answer': 'Start Block'
                })
            else:
                conditions.update({
                    'part': 'End Block',
                    'analysetype': 'End Block',
                    'congruency': 'End Block',
                    'trial': 'End Block',
                    'answer': 'End Block'
                })
            return conditions
        
        # Answer determination (correct, incorrect)
        if event_code < 100:  # Two-digit codes
            if digit2 in [2, 4, 6, 8]:
                conditions['answer'] = 'correct'
            elif digit2 in [3, 5, 7, 9]:
                conditions['answer'] = 'incorrect'
        else:  # Three-digit response codes
            conditions['answer'] = 'incorrect'  # Response codes indicate incorrect answers
        
        # Congruency and trial type (not for fixation)
        if conditions['part'] != 'Fixation' and event_code < 100:
            # Congruency (based on units digit for two-digit codes)
            if digit3 in [1, 3]:
                conditions['congruency'] = 'incongruent'
            elif digit3 in [2, 4]:
                conditions['congruency'] = 'congruent'
            
            # Trial type (based on units digit for two-digit codes)
            if digit3 in [1, 2]:
                conditions['trial'] = 'inducer'
            elif digit3 in [3, 4]:
                conditions['trial'] = 'diagnostic'
        elif conditions['part'] == 'Fixation':
            conditions.update({
                'congruency': 'Fixation',
                'trial': 'Fixation',
                'analysetype': 'Fixation',
                'answer': 'Fixation'
            })
        
        # Analyse type determination (based on tens digit for two-digit codes)
        if event_code < 100:
            if digit2 in [2, 3]:
                conditions['analysetype'] = 'MI'
            elif digit2 in [4, 5]:
                conditions['analysetype'] = 'MC'
            elif digit2 in [6, 7]:
                conditions['analysetype'] = 'main_incon'
            elif digit2 in [8, 9]:
                conditions['analysetype'] = 'main_con'
            
    except (ValueError, IndexError):
        # Keep default values if parsing fails
        pass
    
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
    Create event dictionary for epoching
    
    Returns:
    --------
    event_dict : dict
        Mapping of event names to trigger codes
    """
    event_dict = {
        # MI events
        'MI_correct_inducer_congruent': 21,
        'MI_correct_inducer_incongruent': 23,
        'MI_correct_diagnostic_congruent': 24,
        'MI_correct_diagnostic_incongruent': 22,
        'MI_incorrect_inducer_congruent': 31,
        'MI_incorrect_inducer_incongruent': 33,
        'MI_incorrect_diagnostic_congruent': 34,
        'MI_incorrect_diagnostic_incongruent': 32,
        
        # MC events
        'MC_correct_inducer_congruent': 41,
        'MC_correct_inducer_incongruent': 43,
        'MC_correct_diagnostic_congruent': 44,
        'MC_correct_diagnostic_incongruent': 42,
        'MC_incorrect_inducer_congruent': 51,
        'MC_incorrect_inducer_incongruent': 53,
        'MC_incorrect_diagnostic_congruent': 54,
        'MC_incorrect_diagnostic_incongruent': 52,
        
        # Main incongruent events
        'main_incon_correct_inducer_congruent': 61,
        'main_incon_correct_inducer_incongruent': 63,
        'main_incon_correct_diagnostic_congruent': 64,
        'main_incon_correct_diagnostic_incongruent': 62,
        'main_incon_incorrect_inducer_congruent': 71,
        'main_incon_incorrect_inducer_incongruent': 73,
        'main_incon_incorrect_diagnostic_congruent': 74,
        'main_incon_incorrect_diagnostic_incongruent': 72,
        
        # Main congruent events
        'main_con_correct_inducer_congruent': 81,
        'main_con_correct_inducer_incongruent': 83,
        'main_con_correct_diagnostic_congruent': 84,
        'main_con_correct_diagnostic_incongruent': 82,
        'main_con_incorrect_inducer_congruent': 91,
        'main_con_incorrect_inducer_incongruent': 93,
        'main_con_incorrect_diagnostic_congruent': 94,
        'main_con_incorrect_diagnostic_incongruent': 92,
    }
    
    return event_dict


def get_block_events() -> Dict[str, List[int]]:
    """
    Get block onset and offset trigger codes as integers
    
    Returns:
    --------
    block_events : dict
        Dictionary with block onset and offset trigger codes as integers
    """
    block_events = {
        'onset': [2, 3, 4, 5, 6, 7, 8],      # 'S  2' to 'S  8' -> integers 2-8
        'offset': [101, 102, 103, 104, 105, 106, 107],  # 'S101' to 'S107' -> integers 101-107
        'start_block': 1,    # 'S  1' -> integer 1
        'end_block': 108     # 'S108' -> integer 108
    }
    
    return block_events
