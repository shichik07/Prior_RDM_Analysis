"""Main EEG Preprocessing Pipeline.

Orchestrates the complete preprocessing workflow from raw BIDS data to
final epochs.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mne
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from extract import __version__
from extract.artifacts import apply_asr, apply_line_noise_removal, detect_bad_channels, ica_pipeline
from extract.epoching import final_processing
from extract.events import code_events
from extract.loading import get_subject_list, load_eeg_data, validate_bids_structure
from extract.preprocessing import preprocess_pipeline
from extract.utils import create_config_file, create_output_structure, log_processing_stage


def validate_inputs(bids_root: str, output_dir: str, subject_id: Optional[str] = None) -> None:
    """Validate input parameters and directory structure.

    Args:
        bids_root: Path to BIDS root directory.
        output_dir: Path to output directory.
        subject_id: Specific subject to process, or None to skip subject check.
    """
    if not os.path.exists(bids_root):
        raise FileNotFoundError(f"BIDS root directory not found: {bids_root}")

    if not validate_bids_structure(bids_root):
        raise ValueError("Invalid BIDS directory structure")

    os.makedirs(output_dir, exist_ok=True)

    if subject_id:
        subject_dir = os.path.join(bids_root, subject_id)
        if not os.path.exists(subject_dir):
            raise FileNotFoundError(f"Subject directory not found: {subject_dir}")


def run_single_subject(
    subject_id: str,
    bids_root: str,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Run the complete preprocessing pipeline for a single subject.

    Args:
        subject_id: Subject identifier (e.g. 'sub-01').
        bids_root: Path to BIDS root directory.
        output_dir: Path to output directory.
        config: Optional configuration parameters.

    Returns:
        True if processing completed successfully, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Processing subject: {subject_id}")
    print(f"{'='*60}")

    try:
        # Step 1: Load raw data
        print("Step 1: Loading raw EEG data...")
        raw: mne.io.Raw = load_eeg_data(subject_id, bids_root)
        print(f"  Loaded: {len(raw.times)} samples, {len(raw.ch_names)} channels")

        # Step 2: Event coding
        print("Step 2: Coding events...")
        events: Any
        metadata: pd.DataFrame
        events, metadata = code_events(raw)
        print(f"  Found {len(events)} events")

        # Step 3: Basic preprocessing
        print("Step 3: Basic preprocessing...")
        raw = preprocess_pipeline(raw, metadata)
        print(f"  Preprocessed: {len(raw.times)} samples remaining")

        # Step 4: Channel quality control
        print("Step 4: Channel quality control...")
        rejected_info: Dict[str, List[str]]
        raw, rejected_info = detect_bad_channels(raw)
        print(f"  Bad channels detected: {rejected_info['all']}")

        # Step 5: Line noise removal
        print("Step 5: Line noise removal...")
        raw = apply_line_noise_removal(raw)
        print("  50 Hz line noise removed")

        # Step 6: ASR burst artefact reconstruction
        print("Step 6: ASR burst artefact reconstruction...")
        asr_cutoff: float = config.get("artifacts", {}).get("asr_cutoff", 20.0) if config else 20.0
        raw = apply_asr(raw, cutoff=asr_cutoff)
        print("  ASR reconstruction applied")

        # Step 7: ICA decomposition
        print("Step 7: ICA decomposition...")
        n_components: int = config.get("artifacts", {}).get("n_components", 20) if config else 20
        ica = ica_pipeline(raw, n_components=n_components)
        print(f"  ICA computed: {len(ica.exclude)} bad components auto-detected")

        # Step 8: Final processing and epoching
        print("Step 8: Final processing and epoching...")
        epochs = final_processing(raw, ica, metadata, subject_id, output_dir)
        print(f"  Extracted {len(epochs)} epochs")

        log_processing_stage(
            subject_id,
            "pipeline_completed",
            datetime.now(),
            output_dir,
            n_epochs=len(epochs),
            n_channels=len(epochs.ch_names),
            duration=epochs.tmax - epochs.tmin,
        )

        print(f"✅ {subject_id} processing completed successfully!")
        return True

    except Exception as e:
        import traceback

        print(f"❌ Error processing {subject_id}: {e}")
        traceback.print_exc()
        return False


def run_batch_processing(
    bids_root: str,
    output_dir: str,
    subject_list: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, bool]:
    """Run the preprocessing pipeline for multiple subjects.

    Args:
        bids_root: Path to BIDS root directory.
        output_dir: Path to output directory.
        subject_list: Subject IDs to process, or None to discover from bids_root.
        config: Optional configuration parameters.

    Returns:
        Dict mapping each subject ID to its success status.
    """
    if subject_list is None:
        subject_list = get_subject_list(bids_root)

    print(f"Starting batch processing for {len(subject_list)} subjects")

    results: Dict[str, bool] = {}
    successful: int = 0

    for i, subject_id in enumerate(subject_list, 1):
        print(f"\nProgress: {i}/{len(subject_list)}")
        success: bool = run_single_subject(subject_id, bids_root, output_dir, config)
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
    """Create a default preprocessing configuration.

    Returns:
        Dict of default configuration parameters.
    """
    config: Dict[str, Any] = {
        "preprocessing": {
            "resample_freq": 250,
            "highpass_freq": 0.1,
            "line_noise_freq": 50,
            "reference_channel": "FCz",
        },
        "artifacts": {
            "flatline_threshold": 1e-6,
            "correlation_threshold": 0.8,
            "line_noise_threshold": 5,
            "asr_cutoff": 20.0,
            "n_components": 20,
        },
        "epoching": {
            "tmin": -0.2,
            "tmax": 1.0,
            "baseline": None,
        },
        "pipeline": {
            "skip_ica": False,
            "verbose": True,
        },
    }
    return config


def main() -> None:
    """Entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="EEG Preprocessing Pipeline for Adaptive Control Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python main_pipeline.py --bids-root /path/to/bids --output /path/to/output --subject sub-01
  python main_pipeline.py --bids-root /path/to/bids --output /path/to/output
  python main_pipeline.py --create-config /path/to/config.json
        """,
    )

    parser.add_argument("--bids-root", required=False, help="Path to BIDS root directory")
    parser.add_argument("--output", required=False, help="Path to output directory")
    parser.add_argument("--subject", help="Single subject to process")
    parser.add_argument("--subjects", nargs="+", help="Multiple subjects to process")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--create-config", help="Write default config to this path and exit")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--version", action="version", version=f"EEG Pipeline v{__version__}")

    args = parser.parse_args()

    if args.create_config:
        config = create_default_config()
        create_config_file(os.path.dirname(args.create_config), config)
        print(f"Configuration file created: {args.create_config}")
        return

    if not args.bids_root or not args.output:
        parser.error("--bids-root and --output are required unless using --create-config")

    config = create_default_config()
    if args.config and os.path.exists(args.config):
        from extract.utils import load_config_file

        config = load_config_file(args.config)

    try:
        validate_inputs(args.bids_root, args.output, args.subject)

        if args.subject:
            subject_list: List[str] = [args.subject]
        elif args.subjects:
            subject_list = args.subjects
        else:
            subject_list = get_subject_list(args.bids_root)

        if len(subject_list) == 1:
            run_single_subject(subject_list[0], args.bids_root, args.output, config)
        else:
            run_batch_processing(args.bids_root, args.output, subject_list, config)

    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
