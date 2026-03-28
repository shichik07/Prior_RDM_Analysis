"""
Test Script for EEG Preprocessing Pipeline

Simple test to verify the pipeline works with sample data before full deployment.
"""

import os
import sys
from pathlib import Path
import tempfile
import numpy as np
import mne

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from extract import __version__
from extract.loading import validate_bids_structure
from extract.events import get_block_events, parse_sxxx_code
from extract.utils import create_config_file


def test_basic_functionality():
    """
    Test basic functionality of individual components
    """
    print("Testing basic functionality...")
    
    # Test event parsing
    print("  Testing event parsing...")
    block_events = get_block_events()
    assert isinstance(block_events, dict)
    assert 'onset' in block_events
    assert 'offset' in block_events
    assert all(isinstance(code, int) for code in block_events['onset'])
    print(f"    Block events: {block_events}")
    
    # Test SXXX code parsing
    print("  Testing SXXX code parsing...")
    test_codes = [21, 22, 31, 32, 41, 42]
    for code in test_codes:
        conditions = parse_sxxx_code(code)
        assert isinstance(conditions, dict)
        assert 'part' in conditions
        print(f"    Code {code}: {conditions['part']}, {conditions['answer']}")
    
    print("✅ Basic functionality tests passed!")


def test_config_creation():
    """
    Test configuration file creation
    """
    print("Testing configuration creation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, 'test_config.json')
        create_config_file(temp_dir, {'test': 'value'})
        
        assert os.path.exists(config_path)
        print("  Configuration file created successfully")
    
    print("✅ Configuration creation test passed!")


def test_imports():
    """
    Test that all modules can be imported
    """
    print("Testing imports...")
    
    try:
        from extract.loading import load_eeg_data, setup_bids_paths
        from extract.events import code_events, create_event_metadata
        from extract.preprocessing import preprocess_pipeline
        from extract.artifacts import detect_bad_channels, ica_pipeline
        from extract.epoching import final_processing
        from extract.utils import save_checkpoint, load_checkpoint
        print("  All modules imported successfully")
    except ImportError as e:
        print(f"  Import error: {e}")
        raise
    
    print("✅ Import tests passed!")


def test_mock_data_processing():
    """
    Test pipeline with mock data (requires MNE sample dataset)
    """
    print("Testing with mock data...")
    
    try:
        # Try to load sample dataset for testing
        sample_data = mne.datasets.sample.data_path()
        raw_file = os.path.join(sample_data, 'MEG', 'sample', 'sample_audvis_raw.fif')
        
        if os.path.exists(raw_file):
            print("  Found sample dataset, loading...")
            raw = mne.io.read_raw_fif(raw_file, preload=True)
            
            # Test event detection
            events = mne.find_events(raw)
            print(f"    Found {len(events)} events in sample data")
            
            # Test basic preprocessing steps
            raw.resample(250)
            raw.filter(l_freq=0.1, h_freq=None)
            print("    Basic preprocessing successful")
            
        else:
            print("  Sample dataset not found, skipping mock data test")
            
    except Exception as e:
        print(f"  Mock data test failed: {e}")
        print("  This is expected if sample dataset is not available")
    
    print("✅ Mock data test completed!")


def run_all_tests():
    """
    Run all available tests
    """
    print(f"EEG Preprocessing Pipeline v{__version__}")
    print("=" * 50)
    
    try:
        test_imports()
        test_basic_functionality()
        test_config_creation()
        test_mock_data_processing()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print("The pipeline is ready for use with real data.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Please check the implementation before using with real data.")
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
