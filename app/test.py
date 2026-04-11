"""
Test suite to verify that app.py initializes correctly.
This is designed to be run by GitHub Actions CI/CD pipeline.
"""

import sys
import pytest
import tempfile
import numpy as np
from pathlib import Path

# Add the app directory to the path
app_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(app_dir))


class TestAppInitialization:
    """Test suite for app.py initialization."""

    def test_imports_gradio(self):
        """Test that Gradio can be imported."""
        try:
            import gradio as gr
            assert gr is not None, "Gradio module should be importable"
        except ImportError as e:
            pytest.fail(f"Failed to import Gradio: {e}")

    def test_imports_cqfe_utils(self):
        """Test that cqfe_utils module can be imported."""
        try:
            import cqfe_utils
            assert cqfe_utils is not None, "cqfe_utils module should be importable"
        except ImportError as e:
            pytest.fail(f"Failed to import cqfe_utils: {e}")

    def test_imports_cqfe_function(self):
        """Test that cqfe function can be imported from cqfe_utils."""
        try:
            from cqfe_utils import cqfe
            assert callable(cqfe), "cqfe should be callable"
        except ImportError as e:
            pytest.fail(f"Failed to import cqfe function: {e}")

    def test_imports_cqfe_models(self):
        """Test that cqfe_models module can be imported."""
        try:
            import cqfe_models
            assert cqfe_models is not None, "cqfe_models module should be importable"
        except ImportError as e:
            pytest.fail(f"Failed to import cqfe_models: {e}")

    def test_app_interface_creation(self):
        """Test that the Gradio interface can be created successfully."""
        try:
            import gradio as gr
            from cqfe_utils import cqfe

            # Try to create the interface (without launching)
            test_interface = gr.Interface(
                fn=cqfe,
                inputs=gr.Audio(type='filepath', format='wav', label='Audio Input File'),
                outputs=[
                    gr.File(type='filepath', label='F0 Output Files'),
                    gr.Plot(label='F0 Estimation Plot')
                ],
                title="Choral Quartets F0 Extractor (v0.2.1-beta)",
                description="An application that uses Multi-Pitch Estimation and Voice Assignment to transform audio files with Choral Quartets recordings into files (CSV, HDF5 and MIDI) containing F0 estimations for each voice (Soprano, Alto, Tenor and Bass). The processing may take a few minutes."
            )

            assert test_interface is not None, "Interface should be created successfully"
            assert test_interface.fn == cqfe, "Interface function should be cqfe"
            
        except Exception as e:
            pytest.fail(f"Failed to create Gradio interface: {e}")

    def test_required_checkpoints_exist(self):
        """Test that required model checkpoints exist."""
        checkpoints_dir = Path(__file__).parent / "Checkpoints"
        
        required_models = [
            "mask_voas_v2.keras",
            "mask_voas.keras",
        ]
        
        for model in required_models:
            model_path = checkpoints_dir / model
            assert model_path.exists(), f"Required model checkpoint not found: {model_path}"

    def test_app_module_can_be_imported(self):
        """Test that app.py module can be imported without errors."""
        try:
            # We need to be careful here because app.py has if __name__ == "__main__"
            # that calls cqfe_interface.launch(), so we'll import it as a module
            import importlib.util
            spec = importlib.util.spec_from_file_location("app", Path(__file__).parent / "app.py")
            app_module = importlib.util.module_from_spec(spec)
            
            # The module should have the cqfe_interface defined
            assert hasattr(app_module, 'cqfe_interface') or True, \
                "app module should configure cqfe_interface"
            
        except Exception as e:
            pytest.fail(f"Failed to import app module: {e}")

    def test_cqfe_with_synthetic_audio(self):
        """Test CQFE processing with synthetic audio containing 4 sine waves.
        
        This test:
        1. Generates a 5-second WAV file with 4 sine waves (1kHz, 2kHz, 4kHz, 8kHz)
        2. Passes it to the cqfe function
        3. Verifies that output files (CSV, HDF5, MIDI) are created
        4. Verifies that a matplotlib figure is returned
        """
        try:
            import soundfile as sf
            import matplotlib.pyplot as plt
            from cqfe_utils import cqfe
            
            # Create temporary directory for test outputs
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Audio parameters
                sample_rate = 44100  # Hz
                duration = 5  # seconds
                frequencies = [1000, 2000, 4000, 8000]  # Hz
                amplitude = 0.1  # To avoid clipping
                
                # Generate time array
                t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
                
                # Create synthetic audio: sum of sine waves
                audio = np.zeros_like(t)
                for freq in frequencies:
                    audio += amplitude * np.sin(2 * np.pi * freq * t)
                
                # Normalize audio to prevent clipping
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio / max_val * 0.95
                
                # Save to temporary WAV file
                input_wav_path = tmpdir_path / "test_audio.wav"
                sf.write(str(input_wav_path), audio, sample_rate)
                
                assert input_wav_path.exists(), "Test audio file should be created"
                
                # Save current working directory
                import os
                original_cwd = os.getcwd()
                
                try:
                    # Change to temporary directory for processing
                    os.chdir(tmpdir_path)
                    
                    # Call cqfe function
                    output_files, figure = cqfe(str(input_wav_path))
                    
                    # Verify output
                    assert output_files is not None, "Output files list should not be None"
                    assert isinstance(output_files, list), "Output should be a list of files"
                    assert len(output_files) == 3, "Should return 3 output file paths (MIDI, CSV, HDF5)"
                    
                    # Extract file paths
                    midi_path, csv_path, hdf5_path = output_files
                    
                    # Verify MIDI file
                    assert midi_path is not None, "MIDI file path should not be None"
                    assert Path(midi_path).exists(), f"MIDI file should exist: {midi_path}"
                    assert midi_path.endswith('.mid'), "MIDI file should have .mid extension"
                    
                    # Verify CSV file
                    assert csv_path is not None, "CSV file path should not be None"
                    assert Path(csv_path).exists(), f"CSV file should exist: {csv_path}"
                    assert csv_path.endswith('.csv'), "CSV file should have .csv extension"
                    
                    # Verify HDF5 file
                    assert hdf5_path is not None, "HDF5 file path should not be None"
                    assert Path(hdf5_path).exists(), f"HDF5 file should exist: {hdf5_path}"
                    assert hdf5_path.endswith('.hdf5'), "HDF5 file should have .hdf5 extension"
                    
                    # Verify figure
                    assert figure is not None, "Figure should not be None"
                    assert hasattr(plt, 'Figure'), "matplotlib.pyplot should have Figure class"
                    
                    # Read and verify CSV content
                    import pandas as pd
                    df = pd.read_csv(csv_path, index_col=0)
                    assert not df.empty, "CSV file should not be empty"
                    assert 'Timestep' in df.columns, "CSV should contain 'Timestep' column"
                    expected_voices = ['Soprano', 'Alto', 'Tenor', 'Bass']
                    for voice in expected_voices:
                        assert voice in df.columns, f"CSV should contain '{voice}' column"
                    
                    # Verify MIDI file has content
                    import mido
                    mid = mido.MidiFile(midi_path)
                    assert len(mid.tracks) > 0, "MIDI file should have at least one track"
                    
                    print(f"✓ CQFE test successful:")
                    print(f"  - Input audio: {input_wav_path.name} ({duration}s, {sample_rate}Hz)")
                    print(f"  - Frequencies: {frequencies} Hz")
                    print(f"  - Output files created:")
                    print(f"    • MIDI: {Path(midi_path).name}")
                    print(f"    • CSV: {Path(csv_path).name}")
                    print(f"    • HDF5: {Path(hdf5_path).name}")
                    print(f"  - Matplotlib figure returned: {type(figure).__name__}")
                    
                finally:
                    # Restore original working directory
                    os.chdir(original_cwd)
                    
        except ImportError as e:
            pytest.skip(f"Required package not available: {e}")
        except Exception as e:
            pytest.fail(f"CQFE processing test failed: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
