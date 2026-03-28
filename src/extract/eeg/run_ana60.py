import os
import sys
from pathlib import Path

import mne

# Ensure parent directory is on the path so we can import extract package
sys.path.append(str(Path(__file__).parent.parent))

from extract.events import code_events
from extract.preprocessing import preprocess_pipeline
from extract.artifacts import detect_bad_channels, asr_replacement, ica_pipeline, apply_line_noise_removal
from extract.epoching import final_processing


def main():
    # Paths
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "eeg" / "ANA60"
    vhdr_path = data_dir / "RDM_HC_participant_ANA60.vhdr"

    if not vhdr_path.exists():
        raise FileNotFoundError(f"Could not find ANA60 .vhdr file at {vhdr_path}")

    output_dir = project_root / "derivatives" / "eeg_preproc_ANA60"
    os.makedirs(output_dir, exist_ok=True)

    subject_id = "ANA60"

    print("==============================================")
    print("Running preprocessing pipeline for ANA60")
    print("Input:", vhdr_path)
    print("Output dir:", output_dir)
    print("==============================================")

    # 1) Load raw BrainVision EEG
    print("Step 1: Loading raw EEG data...")
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
    print(f"  Loaded: {len(raw.times)} samples, {len(raw.ch_names)} channels")

    # 2) Event coding
    print("Step 2: Coding events...")
    events, metadata = code_events(raw)
    print(f"  Found {len(events)} events")
    if "part" in metadata.columns:
        print(f"  Event types: {metadata['part'].value_counts().to_dict()}")

    # 3) Basic preprocessing
    print("Step 3: Basic preprocessing...")
    raw = preprocess_pipeline(raw, metadata)
    print(f"  Preprocessed: {len(raw.times)} samples remaining")

    # 4) Channel quality control
    print("Step 4: Channel quality control...")
    raw, rejected_info = detect_bad_channels(raw)
    print(f"  Bad channels detected: {rejected_info['all_bad_channels']}")

    # 5) Line noise removal
    print("Step 5: Line noise removal...")
    raw = apply_line_noise_removal(raw)
    print("  50Hz line noise removed")

    # 6) ASR-like artifact rejection
    print("Step 6: Artifact rejection...")
    raw = asr_replacement(raw)
    print("  ASR-like rejection applied")

    # 7) ICA decomposition
    print("Step 7: ICA decomposition...")
    ica = ica_pipeline(raw, n_components=20)
    print(f"  ICA computed: {len(ica.exclude)} bad components (auto-marked)")

    # 8) Final processing and epoching
    print("Step 8: Final processing and epoching...")
    epochs = final_processing(raw, ica, metadata, subject_id, str(output_dir))
    print(f"  Extracted {len(epochs)} epochs")

    print("==============================================")
    print("ANA60 preprocessing finished.")
    print("Check outputs in:", output_dir)
    print("==============================================")


if __name__ == "__main__":
    main()
