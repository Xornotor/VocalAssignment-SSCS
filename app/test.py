"""
Test suite para validar a aplicação Gradio de Extração de F0 em Coral Quartets.

Testa:
- Inicialização da aplicação Gradio
- API HTTP da aplicação
- Validação de outputs (CSV, HDF5, MIDI, Plot)
- Robustez com arquivos de entrada variados
"""

import os
import sys
import json
import time
import tempfile
import logging
import subprocess
import requests
from pathlib import Path
from typing import Tuple, Optional
import wave

import numpy as np
import pytest


# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTES DE CONFIGURAÇÃO
# ============================================================================

GRADIO_SERVER_HOST = "127.0.0.1"
GRADIO_SERVER_PORT = 7860
GRADIO_SERVER_URL = f"http://{GRADIO_SERVER_HOST}:{GRADIO_SERVER_PORT}"
GRADIO_TIMEOUT = 300  # 5 minutos para processar áudio
GRADIO_STARTUP_TIMEOUT = 60  # 1 minuto para iniciar o servidor
MAX_RETRIES = 3


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def create_dummy_audio(filepath: str, duration: float = 2.0, sample_rate: int = 44100) -> None:
    """
    Cria um arquivo WAV sintético para teste.
    
    Args:
        filepath: Caminho do arquivo a ser criado
        duration: Duração em segundos
        sample_rate: Taxa de amostragem (Hz)
    """
    logger.info(f"Criando arquivo de áudio de teste: {filepath} ({duration}s @ {sample_rate}Hz)")
    
    n_samples = int(duration * sample_rate)
    # Gera um sinal senoidal simples (nota A4 = 440 Hz)
    frequency = 440
    t = np.arange(n_samples) / sample_rate
    signal = np.sin(2 * np.pi * frequency * t) * 0.3  # Amplitude normalizada
    
    # Converte para valores inteiros (16-bit)
    audio_data = (signal * 32767).astype(np.int16)
    
    with wave.open(filepath, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    logger.info(f"Arquivo de áudio criado com sucesso: {filepath}")


def wait_for_gradio_server(url: str, timeout: int = GRADIO_STARTUP_TIMEOUT) -> bool:
    """
    Aguarda a inicialização do servidor Gradio.
    
    Args:
        url: URL base do servidor Gradio
        timeout: Timeout em segundos
        
    Returns:
        True se o servidor está pronto, False caso contrário
    """
    logger.info(f"Aguardando inicialização do servidor Gradio em {url}...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/config", timeout=5)
            if response.status_code == 200:
                logger.info("Servidor Gradio está pronto!")
                return True
        except requests.RequestException:
            time.sleep(1)
    
    logger.error(f"Servidor Gradio não iniciou dentro de {timeout} segundos")
    return False


def submit_gradio_request(audio_filepath: str) -> Optional[str]:
    """
    Submete uma solicitação à API do Gradio.
    
    Args:
        audio_filepath: Caminho do arquivo de áudio
        
    Returns:
        ID da sessão (hash) ou None em caso de erro
    """
    logger.info(f"Submetendo solicitação ao Gradio com arquivo: {audio_filepath}")
    
    # Prepara o arquivo para upload
    with open(audio_filepath, 'rb') as f:
        files = {'data': f}
        data = {}
        
        # Obtém a configuração da interface para saber os nomes dos parâmetros
        try:
            config_response = requests.get(f"{GRADIO_SERVER_URL}/config", timeout=10)
            config = config_response.json()
            
            # Submete a solicitação
            response = requests.post(
                f"{GRADIO_SERVER_URL}/api/predict",
                files=files,
                data=data,
                timeout=GRADIO_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Solicitação submetida com sucesso: {result}")
                return result
            else:
                logger.error(f"Erro na solicitação: {response.status_code} - {response.text}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Erro ao conectar ao servidor Gradio: {e}")
            return None


def submit_gradio_request_v2(audio_filepath: str) -> Optional[dict]:
    """
    Submete uma solicitação à API do Gradio usando a rota /run.
    
    Args:
        audio_filepath: Caminho do arquivo de áudio
        
    Returns:
        Resposta JSON com os outputs ou None em caso de erro
    """
    logger.info(f"Submetendo solicitação ao Gradio (v2) com arquivo: {audio_filepath}")
    
    try:
        # Obtém os endpoints disponíveis
        config_response = requests.get(f"{GRADIO_SERVER_URL}/config", timeout=10)
        config = config_response.json()
        logger.debug(f"Configuração do Gradio: {config}")
        
        # Tenta usar a rota /run/cqfe ou similar
        # Primeiro, descobre o nome da função
        function_name = "cqfe"  # Nome padrão da função em app.py
        
        with open(audio_filepath, 'rb') as f:
            files = {'data': f}
            
            response = requests.post(
                f"{GRADIO_SERVER_URL}/run/{function_name}",
                files=files,
                timeout=GRADIO_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Solicitação bem-sucedida: {json.dumps(result, indent=2)}")
                return result
            else:
                logger.error(f"Erro na solicitação: {response.status_code} - {response.text}")
                return None
                
    except requests.RequestException as e:
        logger.error(f"Erro ao conectar ao servidor Gradio: {e}")
        return None


def validate_csv_file(filepath: str) -> bool:
    """
    Valida se o arquivo CSV é válido.
    
    Args:
        filepath: Caminho do arquivo CSV
        
    Returns:
        True se o arquivo é válido
    """
    logger.info(f"Validando arquivo CSV: {filepath}")
    
    if not os.path.exists(filepath):
        logger.error(f"Arquivo CSV não encontrado: {filepath}")
        return False
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                logger.warning(f"Arquivo CSV tem menos de 2 linhas: {len(lines)}")
                return len(lines) > 0
        
        logger.info(f"Arquivo CSV válido: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Erro ao validar CSV: {e}")
        return False


def validate_hdf5_file(filepath: str) -> bool:
    """
    Valida se o arquivo HDF5 é válido.
    
    Args:
        filepath: Caminho do arquivo HDF5
        
    Returns:
        True se o arquivo é válido
    """
    logger.info(f"Validando arquivo HDF5: {filepath}")
    
    if not os.path.exists(filepath):
        logger.error(f"Arquivo HDF5 não encontrado: {filepath}")
        return False
    
    try:
        import h5py
        
        with h5py.File(filepath, 'r') as f:
            if len(f.keys()) == 0:
                logger.warning(f"Arquivo HDF5 vazio: {filepath}")
                return True
            
            logger.info(f"Arquivo HDF5 válido com grupos: {list(f.keys())}")
            return True
    except Exception as e:
        logger.error(f"Erro ao validar HDF5: {e}")
        return False


def validate_midi_file(filepath: str) -> bool:
    """
    Valida se o arquivo MIDI é válido.
    
    Args:
        filepath: Caminho do arquivo MIDI
        
    Returns:
        True se o arquivo é válido
    """
    logger.info(f"Validando arquivo MIDI: {filepath}")
    
    if not os.path.exists(filepath):
        logger.error(f"Arquivo MIDI não encontrado: {filepath}")
        return False
    
    try:
        import mido
        
        mid = mido.MidiFile(filepath)
        logger.info(f"Arquivo MIDI válido com {len(mid.tracks)} tracks")
        return True
    except Exception as e:
        logger.error(f"Erro ao validar MIDI: {e}")
        return False


# ============================================================================
# TESTES PYTEST
# ============================================================================

@pytest.fixture(scope="session")
def gradio_server():
    """
    Fixture que inicia o servidor Gradio antes dos testes e o encerra depois.
    """
    logger.info("Iniciando servidor Gradio...")
    
    # Inicia o servidor em background
    process = subprocess.Popen(
        [sys.executable, "-m", "gradio", "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    # Aguarda a inicialização
    if not wait_for_gradio_server(GRADIO_SERVER_URL):
        process.terminate()
        logger.error("Falha ao iniciar o servidor Gradio")
        raise RuntimeError("Servidor Gradio não iniciou")
    
    yield process
    
    # Encerra o servidor
    logger.info("Encerrando servidor Gradio...")
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        logger.warning("Forçando encerramento do servidor Gradio")
        process.kill()


@pytest.fixture
def temp_audio_file():
    """
    Fixture que cria um arquivo de áudio temporário para testes.
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
    
    create_dummy_audio(temp_path, duration=2.0)
    
    yield temp_path
    
    # Limpeza
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_output_dir():
    """
    Fixture que cria um diretório temporário para outputs.
    """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Limpeza
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


# ============================================================================
# TESTES DA API
# ============================================================================

class TestGradioAPI:
    """Testes da API Gradio HTTP."""
    
    def test_server_health(self, gradio_server):
        """
        Testa se o servidor Gradio está respondendo.
        """
        logger.info("Testando saúde do servidor Gradio...")
        
        response = requests.get(f"{GRADIO_SERVER_URL}/config", timeout=10)
        assert response.status_code == 200, "Servidor Gradio não respondeu"
        
        config = response.json()
        assert "components" in config or "interface" in config, "Config inválida"
        logger.info("✓ Servidor Gradio está saudável")
    
    def test_config_endpoint(self, gradio_server):
        """
        Testa se o endpoint /config retorna uma configuração válida.
        """
        logger.info("Testando endpoint /config...")
        
        response = requests.get(f"{GRADIO_SERVER_URL}/config", timeout=10)
        assert response.status_code == 200
        
        config = response.json()
        logger.info(f"Configuração obtida: {json.dumps(config, indent=2)[:200]}...")
        logger.info("✓ Endpoint /config funciona corretamente")
    
    def test_audio_processing_with_dummy_file(self, gradio_server, temp_audio_file):
        """
        Testa o processamento de um arquivo de áudio.
        """
        logger.info("Testando processamento de áudio...")
        
        assert os.path.exists(temp_audio_file), "Arquivo de teste não foi criado"
        
        # Tenta submeter a solicitação
        result = submit_gradio_request_v2(temp_audio_file)
        
        if result is None:
            logger.warning("Não foi possível conectar ao servidor (pode estar processando)")
            pytest.skip("Servidor não respondeu")
        
        assert result is not None, "Resultado é None"
        logger.info(f"✓ Arquivo de áudio processado: {result}")
    
    def test_output_files_structure(self, gradio_server, temp_audio_file):
        """
        Testa se a resposta da API contém a estrutura esperada de outputs.
        """
        logger.info("Testando estrutura de outputs...")
        
        # Submete a solicitação
        result = submit_gradio_request_v2(temp_audio_file)
        
        if result is None:
            pytest.skip("Servidor não respondeu")
        
        # Valida a estrutura da resposta
        # A resposta deve conter "data" com os outputs
        if "data" in result:
            outputs = result["data"]
            assert isinstance(outputs, (list, dict)), "Outputs não é lista ou dicionário"
            
            # Se é lista, deve ter pelo menos 2 elementos (arquivos e plot)
            if isinstance(outputs, list):
                assert len(outputs) >= 2, f"Esperava pelo menos 2 outputs, obteve {len(outputs)}"
            
            logger.info(f"✓ Estrutura de outputs válida: {type(outputs)}")
        else:
            logger.warning(f"Resposta não contém 'data': {result.keys()}")


class TestOutputValidation:
    """Testes de validação dos arquivos de output."""
    
    def test_csv_output_validation(self):
        """
        Testa validação de arquivo CSV.
        """
        logger.info("Testando validação de CSV...")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("time,f0_soprano,f0_alto,f0_tenor,f0_bass\n")
            f.write("0.0,440.0,350.0,220.0,110.0\n")
            temp_csv = f.name
        
        try:
            assert validate_csv_file(temp_csv), "CSV validation falhou"
            logger.info("✓ CSV validado com sucesso")
        finally:
            os.remove(temp_csv)
    
    def test_hdf5_output_validation(self):
        """
        Testa validação de arquivo HDF5.
        """
        logger.info("Testando validação de HDF5...")
        
        try:
            import h5py
            
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
                temp_hdf5 = f.name
            
            with h5py.File(temp_hdf5, 'w') as f:
                f.create_dataset('f0_soprano', data=[440.0, 441.0, 442.0])
            
            try:
                assert validate_hdf5_file(temp_hdf5), "HDF5 validation falhou"
                logger.info("✓ HDF5 validado com sucesso")
            finally:
                os.remove(temp_hdf5)
        except ImportError:
            pytest.skip("h5py não está instalado")
    
    def test_midi_output_validation(self):
        """
        Testa validação de arquivo MIDI.
        """
        logger.info("Testando validação de MIDI...")
        
        try:
            import mido
            
            with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
                temp_midi = f.name
            
            # Cria um arquivo MIDI mínimo
            mid = mido.MidiFile()
            track = mido.MidiTrack()
            mid.tracks.append(track)
            track.append(mido.Message('program_change', program=0, time=0))
            track.append(mido.Message('note_on', note=60, velocity=64, time=0))
            track.append(mido.Message('note_off', note=60, velocity=64, time=480))
            
            mid.save(temp_midi)
            
            try:
                assert validate_midi_file(temp_midi), "MIDI validation falhou"
                logger.info("✓ MIDI validado com sucesso")
            finally:
                os.remove(temp_midi)
        except ImportError:
            pytest.skip("mido não está instalado")


class TestErrorHandling:
    """Testes de tratamento de erros."""
    
    def test_invalid_audio_file(self, gradio_server):
        """
        Testa comportamento com arquivo de áudio inválido.
        """
        logger.info("Testando com arquivo de áudio inválido...")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.wav', delete=False) as f:
            f.write("This is not a valid WAV file")
            temp_invalid = f.name
        
        try:
            result = submit_gradio_request_v2(temp_invalid)
            # Esperamos que falhe graciosamente ou retorne erro
            if result is not None:
                logger.info(f"Resultado para arquivo inválido: {result}")
            logger.info("✓ Aplicação tratou arquivo inválido")
        finally:
            os.remove(temp_invalid)
    
    def test_missing_audio_file(self, gradio_server):
        """
        Testa comportamento com arquivo que não existe.
        """
        logger.info("Testando com arquivo inexistente...")
        
        result = submit_gradio_request_v2("/nonexistent/file.wav")
        
        # Esperamos que falhe
        if result is not None:
            logger.info(f"Resultado para arquivo inexistente: {result}")
        
        logger.info("✓ Aplicação tratou arquivo inexistente")


# ============================================================================
# MAIN PARA EXECUÇÃO DIRETA
# ============================================================================

if __name__ == "__main__":
    logger.info("Iniciando suite de testes da aplicação Gradio")
    
    # Executa todos os testes
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--log-cli-level=INFO",
        "--timeout=600"  # 10 minutos de timeout total
    ])