import gradio as gr
from cqfe_utils import cqfe
cqfe_interface = gr.Interface(fn=cqfe,
                              inputs=gr.Audio(type='filepath', format='wav', label='Audio Input File'),
                              outputs=[gr.File(type='filepath', label='F0 Output Files'),
                                       gr.Plot(label='F0 Estimation Plot')],
                              title="Choral Quartets F0 Extractor (v0.3.0)",
                              description="An application that uses Multi-Pitch Estimation and Voice Assignment to transform audio files with Choral Quartets recordings into files (CSV, HDF5 and MIDI) containing F0 estimations for each voice (Soprano, Alto, Tenor and Bass). The processing may take a few minutes.")

if __name__ == "__main__":
    cqfe_interface.launch()