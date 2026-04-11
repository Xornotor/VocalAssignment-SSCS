import os
import pytest
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

# 1. Mock global do Keras para evitar erro de FileNotFound nos diretórios de Checkpoint
# Isso DEVE ocorrer antes da importação do app.py
import tensorflow as tf
tf.keras.Model.load_weights = MagicMock()

# 2. Agora é seguro importar a interface do Gradio
from app import cqfe_interface

@pytest.fixture
def mock_cqfe():
    """
    Simula a função de processamento pesado para evitar a execução de
    inferência e processamento de áudio durante os testes da API.
    """
    with patch('app.cqfe') as mock:
        # Gera saídas falsas consistentes com a assinatura da função original
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [100, 200, 300]) # Gráfico genérico
        
        # A função original retorna: [mid, csv, hdf5], fig
        mock.return_value = (["output.mid", "output.csv", "output.hdf5"], fig)
        yield mock

def test_gradio_api_endpoint(mock_cqfe, tmp_path):
    """
    Testa se a interface Gradio recebe o áudio e mapeia os outputs corretamente.
    """
    # Cria um arquivo de áudio temporário falso para o teste
    dummy_audio = tmp_path / "test_audio.wav"
    dummy_audio.touch()

    # No Gradio, podemos invocar a interface diretamente como uma função
    result = cqfe_interface(str(dummy_audio))

    # Verifica se a função subjacente (simulada) foi chamada com o caminho do áudio
    mock_cqfe.assert_called_once_with(str(dummy_audio))
    
    # Verifica a estrutura da saída (Gradio retorna uma tupla com as saídas definidas)
    assert result is not None, "A API não deve retornar None"
    assert len(result) == 2, "A API deve retornar dois blocos: [Arquivos] e [Plot]"
    
    arquivos_gerados = result[0]
    plot_gerado = result[1]
    
    # Verifica a geração da lista de arquivos
    assert isinstance(arquivos_gerados, list), "O primeiro output deve ser uma lista de arquivos"
    assert len(arquivos_gerados) == 3, "Devem ser gerados 3 arquivos (MIDI, CSV, HDF5)"
    assert "output.mid" in arquivos_gerados[0]
    
    # Verifica se o componente de Plot não está vazio (Gradio converte plots internamente)
    assert plot_gerado is not None, "O gráfico F0 deve ser gerado e retornado pela API"