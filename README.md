# **Vocal Assignment**

### **Hiperparâmetros e dados de treino**

| Arquitetura           | Treino | Epochs | Alpha | Batch Size | Split Size | Tempo de treino | Tempo por minibatch | Tensorboard | Early Stopping | Train | Dev | Test |
|-----------------------|--------|--------|-------|------------|------------|-----------------|---------------------|-------------|----------------|-------|-----|------|
| MaskVoasCNN           |   #1   | 33     | 2e-3  | 24         | 256        | 4 horas         |                     | Não         | Não            | 1000  | 300 | 805  |
| MaskVoasCNN           |   #2   | 30     | 2e-3  | 24         | 256        | 4 horas         |                     | Sim         | Não            | 1000  | 300 | 805  |
| MaskVoasCNN           |   #3   | 30     | 2e-3  | 16         | 256        | 4 horas         |                     | Sim         | Não            | 1000  | 300 | 805  |
| MaskVoasCNN           |   #4   | 30     | 5e-3  | 16         | 256        | 4 horas         |                     | Sim         | Não            | 1000  | 300 | 805  |
| MaskVoasCNN           |   #5   | 30     | 1e-3  | 24         | 256        | 4 horas         |                     | Sim         | Não            | 1000  | 300 | 805  |
| MaskVoasCNN           |   #6   | 50     | 2e-3  | 24         | 256        | 6,5 horas       |                     | Sim         | Não            | 1000  | 300 | 805  |
| MaskVoasCNNv2         |   #1   | 30     | 5e-3  | 24         | 256        | 4 horas         |                     | Sim         | Não            | 1000  | 300 | 805  |
| DownsampleVoasCNN     |   #1   | 30     | 5e-3  | 24         | 256        | 4 horas         |                     | Sim         | Não            | 1000  | 300 | 805  |
| DownsampleVoasCNNv2   |   #1   | 30     | 5e-3  | 24         | 256        | 4 horas         |                     | Sim         | Não            | 1000  | 300 | 805  |
| VoasCNN (original)    |   #1   | 30     | 5e-3  | 24         | 256        | 12 horas        | 664 ms/step         | Sim         | Não            | 1000  | 300 | 805  |
