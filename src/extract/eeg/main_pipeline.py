"""
Main EEG Preprocessing Pipeline

Orchestrates the complete preprocessing workflow from raw BIDS data to final epochs.
This script provides both individual step processing and full pipeline execution.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import mne
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from extract.loading import load_eeg_data, validate_bids_structure, get_subject_list
from extract.events import code_events
from extract.preprocessing import preprocess_pipeline
from extract.artifacts import detect_bad_channels, asr_replacement, ica_pipeline, apply_line_noise_removal
from extract.epoching import final_processing
from extract.utils import create_output_structure, log_processing_stage, create_config_file
from extract import __version__


def validate_inputs(bids_root: str, output_dir: str, subject_id: Optional[str] = None) -> None:
    """
    Validate input parameters and directory structure
    
    Parameters:
    -----------
    bids_root : str
        Path to BIDS root directory
    output_dir : str
        Path to output directory
    subject_id : str, optional
        Specific subject to process
    """
    if not os.path.exists(bids_root):
        raise FileNotFoundError(f"BIDS root directory not found: {bids_root}")
    
    if not validate_bids_structure(bids_root):
        raise ValueError("Invalid BIDS directory structure")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if subject_id:
        subject_dir = os.path.join(bids_root, subject_id)
        if not os.path.exists(subject_dir):
            raise FileNotFoundError(f"Subject directory not found: {subject_dir}")


def run_single_subject(subject_id: str, bids_root: str, output_dir: str, 
                      config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Run complete preprocessing pipeline for a single subject
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier (e.g., 'sub-01')
    bids_root : str
        Path to BIDS root directory
    output_dir : str
        Path to output directory
    config : dict, optional
        Configuration parameters
    
    Returns:
    --------
    success : bool
        True if processing completed successfully
    """
    from datetime import datetime
    
    print(f"\n{'='*60}")
    print(f"Processing subject: {subject_id}")
    print(f"{'='*60}")
    
    try:
        # Step 1: Load raw data
        print("Step 1: Loading raw EEG data...")
        raw = load_eeg_data(subject_id, bids_root)
        print(f"  Loaded: {len(raw.times)} samples, {len(raw.ch_names)} channels")
        
        # Step 2: Event coding
        print("Step 2: Coding events...")
        events, metadata = code_events(raw)
        print(f"  Found {len(events)} events")
        print(f"  Event types: {metadata['part'].value_counts().to_dict()}")
        
        # Step 3: Basic preprocessing
        print("Step 3: Basic preprocessing...")
        raw = preprocess_pipeline(raw, metadata)
        print(f"  Preprocessed: {len(raw.times)} samples remaining")
        
        # Step 4: Channel quality control
        print("Step 4: Channel quality control...")
        raw, rejected_info = detect_bad_channels(raw)
        print(f"  Bad channels detected: {rejected_info['all_bad_channels']}")
        
        # Step 5: Line noise removal
        print("Step 5: Line noise removal...")
        raw = apply_line_noise_removal(raw)
        print("  50Hz line noise removed")
        
        # Step 6: ASR-like artifact rejection
        print("Step 6: Artifact rejection...")
        raw = asr_replacement(raw)
        print("  ASR-like rejection applied")
        
        # Step 7: ICA decomposition
        print("Step 7: ICA decomposition...")
        n_components = config.get('n_components', 20) if config else 20
        ica = ica_pipeline(raw, n_components=n_components)
        print(f"  ICA computed: {len(ica.exclude)} bad components")
        
        # Step 8: Final processing and epoching
        print("Step 8: Final processing and epoching...")
        epochs = final_processing(raw, ica, metadata, subject_id, output_dir)
        print(f"  Extracted {len(epochs)} epochs")
        
        # Log completion
        log_processing_stage(
            subject_id, 
            "pipeline_completed", 
            datetime.now(),
            output_dir,
            n_epochs=len(epochs),
            n_channels=len(epochs.ch_names),
            duration=epochs.tmax - epochs.tmin
        )
        
        print(f"✅ {subject_id} processing completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {subject_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_batch_processing(bids_root: str, output_dir: str, 
                        subject_list: Optional[list] = None,
                        config: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
    """
    Run preprocessing pipeline for multiple subjects
    
    Parameters:
    -----------
    bids_root : str
        Path to BIDS root directory
    output_dir : str
        Path to output directory
    subject_list : list, optional
        List of subject IDs to process
    config : dict, optional
        Configuration parameters
    
    Returns:
    --------
    results : dict
        Dictionary mapping subject IDs to success status
    """
    if subject_list is None:
        subject_list = get_subject_list(bids_root)
    
    print(f"Starting batch processing for {len(subject_list)} subjects")
    
    results = {}
    successful = 0
    
    for i, subject_id in enumerate(subject_list, 1):
        print(f"\nProgress: {i}/{len(subject_list)}")
        
        success = run_single_subject(subject_id, bids_root, output_dir, config)
        results[subject_id] = success
        
        if success:
            successful += 1
    
    print(f"\n{'='*60}")
    print(f"Batch processing completed!")
    print(f"Successful: {successful}/{len(subject_list)}")
    print(f"Failed: {len(subject_list) - successful}/{len(subject_list)}")
    print(f"{'='*60}")
    
    return results


def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration for preprocessing
    
    Returns:
    --------
    config : dict
        Default configuration parameters
    """
    config = {
        'preprocessing': {
            'resample_freq': 250,
            'highpass_freq': 0.1,
            'line_noise_freq': 50,
            'reference_channel': 'FCz'
        },
        'artifacts': {
            'flatline_threshold': 1e-6,
            'correlation_threshold': 0.8,
            'line_noise_threshold': 5,
            'n_components': 20
        },
        'epoching': {
            'tmin': -0.2,
            'tmax': 1.0,
            'baseline': None
        },
        'pipeline': {
            'skip_ica': False,
            'skip_asr': False,
            'verbose': True
        }
    }
    
    return config


def main():
    """
    Main function for command-line usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='EEG Preprocessing Pipeline for Adaptive Control Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Process single subject
  python main_pipeline.py --bids-root /path/to/bids --output /path/to/output --subject sub-01
  
  # Process all subjects
  python main_pipeline.py --bids-root /path/to/bids --output /path/to/output
  
  # Process specific subjects
  python main_pipeline.py --bids-root /path/to/bids --output /path/to/output --subjects sub-01 sub-02 sub-03
  
  # Create config file
  python main_pipeline.py --create-config /path/to/config.json
        """
    )
    
    parser.add_argument('--bids-root', required=False, help='Path to BIDS root directory')
    parser.add_argument('--output', required=False, help='Path to output directory')
    parser.add_argument('--subject', help='Single subject to process')
    parser.add_argument('--subjects', nargs='+', help='Multiple subjects to process')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--create-config', help='Create default config file at specified path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--version', action='version', version=f'EEG Pipeline v{__version__}')
    
    args = parser.parse_args()
    
    # Handle config creation
    if args.create_config:
        config = create_default_config()
        create_config_file(os.path.dirname(args.create_config), config)
        print(f"Configuration file created: {args.create_config}")
        return
    
    # Validate required arguments
    if not args.bids_root or not args.output:
        parser.error("--bids-root and --output are required unless using --create-config")
    
    # Load configuration
    config = create_default_config()
    if args.config and os.path.exists(args.config):
        from extract.utils import load_config_file
        config = load_config_file(args.config)
    
    try:
        # Validate inputs
        validate_inputs(args.bids_root, args.output, args.subject)
        
        # Determine subjects to process
        if args.subject:
            subject_list = [args.subject]
        elif args.subjects:
            subject_list = args.subjects
        else:
            subject_list = get_subject_list(args.bids_root)
        
        # Run processing
        if len(subject_list) == 1:
            run_single_subject(subject_list[0], args.bids_root, args.output, config)
        else:
            run_batch_processing(args.bids_root, args.output, subject_list, config)
            
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
