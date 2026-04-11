import pytest
import os
import wave
from unittest.mock import patch

# Assuming your Gradio code is in a file named `app.py`
import app 

@pytest.fixture
def dummy_wav_file(tmp_path):
    """Creates a temporary, 1-second silent WAV file for testing inputs."""
    file_path = tmp_path / "test_audio.wav"
    with wave.open(str(file_path), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(44100)
        wav_file.writeframes(b'\x00\x00' * 44100)
    return str(file_path)

# Mock the underlying processing function to avoid long execution times in CI
@patch('app.cqfe') 
def test_gradio_interface(mock_cqfe, dummy_wav_file, tmp_path):
    # 1. Setup the mock return values (A dummy file path and a dummy plot object)
    mock_output_file = tmp_path / "f0_estimations.zip"
    mock_output_file.write_text("dummy data")
    
    mock_cqfe.return_value = (str(mock_output_file), "Mock Plot Object")

    # 2. Extract the processing function bound to the Gradio Interface
    interface_fn = app.cqfe_interface.fn

    # 3. Execute the function with our dummy input
    result_file, result_plot = interface_fn(dummy_wav_file)

    # 4. Assertions to ensure the interface behaves correctly
    mock_cqfe.assert_called_once_with(dummy_wav_file)
    assert result_file == str(mock_output_file)
    assert result_plot == "Mock Plot Object"

def test_gradio_interface_properties():
    """Validates that the Gradio interface is configured with the correct UI elements."""
    assert app.cqfe_interface.title == "Choral Quartets F0 Extractor (v0.2.1-beta)"
    assert len(app.cqfe_interface.input_components) == 1
    assert len(app.cqfe_interface.output_components) == 2
    assert app.cqfe_interface.input_components[0].__class__.__name__ == "Audio"