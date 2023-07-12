# **Vocal Assignment**

### **Hiperparâmetros e dados de treino**

| Arquitetura           | Treino | Epochs | Alpha | Batch Size | Split Size | Tempo de treino | Tensorboard | Train | Dev | Test |
|-----------------------|--------|--------|-------|------------|------------|-----------------|-------------|-------|-----|------|
| MaskVoasCNN           |   #1   | 33     | 2e-3  | 24         | 256        | 4 horas         | Não         | 1000  | 300 | 300  |
| MaskVoasCNN           |   #2   | 30     | 2e-3  | 24         | 256        | 4 horas         | Sim         | 1000  | 300 | 300  |
| MaskVoasCNN           |   #3   | 30     | 2e-3  | 16         | 256        | 4 horas         | Sim         | 1000  | 300 | 300  |
| MaskVoasCNN           |   #4   | 30     | 5e-3  | 16         | 256        | 4 horas         | Sim         | 1000  | 300 | 300  |
| MaskVoasCNN           |   #5   | 30     | 1e-3  | 24         | 256        | 4 horas         | Sim         | 1000  | 300 | 300  |
| MaskVoasCNN           |   #6   | 50     | 2e-3  | 24         | 256        | 6,5 horas       | Sim         | 1000  | 300 | 300  |
| MaskVoasCNNv2         |   #1   | 30     | 5e-3  | 24         | 256        | 4 horas         | Sim         | 1000  | 300 | 300  |
| DownsampleVoasCNN     |   #1   | 30     | 5e-3  | 24         | 256        | 4 horas         | Sim         | 1000  | 300 | 300  |
| DownsampleVoasCNNv2   |   #1   | 30     | 5e-3  | 24         | 256        | 4 horas         | Sim         | 1000  | 300 | 300  |
