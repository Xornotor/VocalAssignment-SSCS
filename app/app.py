import tensorflow as tf
import gradio as gr
from cqfe_utils import cqfe

if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    cqfe_interface = gr.Interface(fn=cqfe,
                            inputs=gr.Audio(type='filepath', format='wav', label='Audio Input File'),
                            outputs=[gr.File(type='filepath', label='F0 Output Files'),
                                    gr.Plot(label='F0 Estimation Plot')],
                            title="Choral Quartets F0 Extractor (v0.4.0)",
                            description="An application that uses Multi-Pitch Estimation and Voice Assignment to transform audio files with Choral Quartets recordings into files (CSV, HDF5 and MIDI) containing F0 estimations for each voice (Soprano, Alto, Tenor and Bass). The processing may take a few minutes.")


    cqfe_interface.launch(share=True)