# **Vocal Assignment**

## **Hiperparâmetros e dados de treino**

### **Holdout**

| Arquitetura           | Treino | Epochs | Alpha | Batch Size | Split Size | Tempo de treino | Tensorboard | Early Stopping | Train | Dev | Test |
|-----------------------|--------|--------|-------|------------|------------|-----------------|-------------|----------------|-------|-----|------|
| MaskVoasCNN           |   #1   | 33     | 2e-3  | 24         | 256        | 4 horas         | Não         | Não            | 1000  | 300 | 805  |
| MaskVoasCNN           |   #2   | 30     | 2e-3  | 24         | 256        | 4 horas         | Sim         | Não            | 1000  | 300 | 805  |
| MaskVoasCNN           |   #3   | 30     | 2e-3  | 16         | 256        | 4 horas         | Sim         | Não            | 1000  | 300 | 805  |
| MaskVoasCNN           |   #4   | 30     | 5e-3  | 16         | 256        | 4 horas         | Sim         | Não            | 1000  | 300 | 805  |
| MaskVoasCNN           |   #5   | 30     | 1e-3  | 24         | 256        | 4 horas         | Sim         | Não            | 1000  | 300 | 805  |
| MaskVoasCNN           |   #6   | 50     | 2e-3  | 24         | 256        | 6,5 horas       | Sim         | Não            | 1000  | 300 | 805  |
| MaskVoasCNNv2         |   #1   | 30     | 5e-3  | 24         | 256        | 4 horas         | Sim         | Não            | 1000  | 300 | 805  |
| DownsampleVoasCNN     |   #1   | 30     | 5e-3  | 24         | 256        | 4 horas         | Sim         | Não            | 1000  | 300 | 805  |
| DownsampleVoasCNNv2   |   #1   | 30     | 5e-3  | 24         | 256        | 4 horas         | Sim         | Não            | 1000  | 300 | 805  |
| VoasCNN (original)    |   #1   | 30     | 5e-3  | 24         | 256        | 12 horas        | Sim         | Não            | 1000  | 300 | 805  |

### **K-Fold (MaskVoasCNNv2)**

| Treino | Fold | Epochs  | Alpha | Batch Size | Split Size | Tempo de treino | Tensorboard | Early Stopping | Train | Dev | Test |
|--------|------|---------|-------|------------|------------|-----------------|-------------|----------------|-------|-----|------|
|   #1   |  #1  | 50 (12) | 5e-3  | 24         | 256        | 4 horas         | Sim         | Sim (6 epochs) | 3200  | 800 | 805  |
|   #1   |  #2  | 50 (9)  | 5e-3  | 24         | 256        | 3 horas         | Sim         | Sim (6 epochs) | 3200  | 800 | 805  |
|   #1   |  #3  | 50 (9)  | 5e-3  | 24         | 256        | 3 horas         | Sim         | Sim (6 epochs) | 3200  | 800 | 805  |
|   #1   |  #4  | 50 (10) | 5e-3  | 24         | 256        | 3,25 horas      | Sim         | Sim (6 epochs) | 3200  | 800 | 805  |
|   #1   |  #5  | 50 (9)  | 5e-3  | 24         | 256        | 4 horas         | Sim         | Sim (6 epochs) | 3200  | 800 | 805  |
|   #2   |  #1  | 50 (23) | 5e-4  | 24         | 256        | 8 horas         | Sim         | Sim (6 epochs) | 3200  | 800 | 805  |
|   #2   |  #2  | 50 (12) | 5e-4  | 24         | 256        | 4 horas         | Sim         | Sim (6 epochs) | 3200  | 800 | 805  |
|   #2   |  #3  | 50 (10) | 5e-4  | 24         | 256        | 3 horas         | Sim         | Sim (6 epochs) | 3200  | 800 | 805  |
|   #2   |  #4  | 50 (17) | 5e-4  | 24         | 256        | 5,75 horas      | Sim         | Sim (6 epochs) | 3200  | 800 | 805  |
|   #2   |  #5  | 50      | 5e-4  | 24         | 256        | X horas         | Sim         | Sim (6 epochs) | 3200  | 800 | 805  |