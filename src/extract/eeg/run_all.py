#!/usr/bin/env python3
"""
Batch runner for PREP-01 EEG preprocessing pipeline.

Usage
-----
# All subjects found in EEG_DIR (default):
    python src/extract/eeg/run_all.py

# One specific subject:
    python src/extract/eeg/run_all.py --subject ANA60

# Explicit list:
    python src/extract/eeg/run_all.py --subject ANA60 ANA61 ANA62

# Dry-run (list subjects, do nothing):
    python src/extract/eeg/run_all.py --dry-run

# Skip already-processed subjects:
    python src/extract/eeg/run_all.py --skip-done
"""

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — keep in sync with run_pipeline.py 
# ---------------------------------------------------------------------------
EEG_DIR    = Path("/mnt/e/priorRDM/Study/EEGData")  # WSL path to external drive
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "processed" / "eeg"


def discover_subjects() -> list[str]:
    """Return sorted list of subject IDs (subfolder names) in EEG_DIR."""
    if not EEG_DIR.exists():
        print(f"[ERROR] EEG data directory not found: {EEG_DIR}")
        print("        Is the external drive (E:) connected?")
        sys.exit(1)
    subjects = sorted(
        d.name for d in EEG_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if not subjects:
        print(f"[ERROR] No subject folders found in {EEG_DIR}")
        sys.exit(1)
    return subjects


def already_processed(subject_id: str) -> bool:
    """True if the expected output epochs file exists for this subject."""
    out = OUTPUT_DIR / subject_id / f"{subject_id}_epochs_stimulus.fif"
    return out.exists()


def run(subjects: list[str], skip_done: bool, dry_run: bool) -> None:
    # Import here so a missing dependency only breaks at runtime, not at import
    from run_pipeline import run_pipeline  # noqa: E402  (same package)

    total   = len(subjects)
    done    = 0
    skipped = 0
    failed  = []

    print("=" * 60)
    print(f"  Prior RDM — EEG batch preprocessing")
    print(f"  Subjects : {total}")
    print(f"  EEG dir  : {EEG_DIR}")
    print(f"  Out dir  : {OUTPUT_DIR}")
    print(f"  Options  : skip_done={skip_done}  dry_run={dry_run}")
    print("=" * 60)

    for i, subj in enumerate(subjects, 1):
        tag = f"[{i}/{total}]"

        if skip_done and already_processed(subj):
            print(f"\n{tag} {subj}  — already processed, skipping.")
            skipped += 1
            continue

        if dry_run:
            print(f"{tag} {subj}  — (dry-run, would process)")
            continue

        print(f"\n{tag} Starting {subj}  ({datetime.now():%H:%M:%S})")
        print("-" * 60)
        try:
            run_pipeline(subj)
            done += 1
        except Exception as exc:  # noqa: BLE001
            print(f"\n[FAILED] {subj}: {exc}")
            traceback.print_exc()
            failed.append((subj, str(exc)))

    # Summary
    print("\n" + "=" * 60)
    if dry_run:
        print(f"  Dry-run complete — {total} subject(s) would be processed.")
    else:
        print(f"  Batch complete")
        print(f"    Processed : {done}")
        print(f"    Skipped   : {skipped}  (already done)")
        print(f"    Failed    : {len(failed)}")
        if failed:
            print("\n  Failed subjects:")
            for subj, err in failed:
                print(f"    {subj}: {err}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-run PREP-01 EEG pipeline across subjects.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--subject", nargs="*", metavar="ID",
        help="One or more subject IDs. Omit to process all subjects in EEG_DIR.",
    )
    parser.add_argument(
        "--skip-done", action="store_true",
        help="Skip subjects whose output epochs file already exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List subjects that would be processed without running anything.",
    )
    args = parser.parse_args()

    if args.subject:
        subjects = args.subject
    else:
        subjects = discover_subjects()

    run(subjects, skip_done=args.skip_done, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
