# -*- coding: utf-8 -*-
# Importamos las bibliotecas necesarias para las pruebas.
import pytest  # El framework de pruebas.
import numpy as np  # Para crear datos de audio falsos.
from unittest.mock import MagicMock, patch  # Para simular ('mockear') la biblioteca TTS.
import os  # Para manejar rutas de archivos.
from scipy.io.wavfile import read as read_wav # Para leer archivos WAV y verificar su contenido.

# Importamos la clase que vamos a probar.
from voxflow_core import Synthesizer

# --- Pruebas para la clase Synthesizer ---

# Usamos el decorador 'patch' para reemplazar temporalmente la clase TTS de la biblioteca Coqui
# por un objeto simulado (MagicMock) durante la ejecución de esta prueba.
@patch('voxflow_core.TTS')
def test_synthesizer_initialization(mock_tts_class):
    """
    # Prueba que la clase Synthesizer se inicializa correctamente.
    # Verifica que el modelo TTS se carga y que la lista de locutores está disponible.
    """
    # -- Arrange (Preparación) --
    # Configuramos el objeto simulado de TTS.
    mock_tts_instance = MagicMock()
    # Definimos una lista falsa de locutores que el objeto simulado devolverá.
    fake_speakers = ["speaker_1", "speaker_2"]
    # Hacemos que la propiedad 'speakers' del objeto simulado devuelva nuestra lista falsa.
    mock_tts_instance.speakers = fake_speakers
    # Configuramos el constructor de la clase TTS simulada para que devuelva nuestra instancia simulada.
    mock_tts_class.return_value.to.return_value = mock_tts_instance

    # -- Act (Acción) --
    # Creamos una instancia de nuestra clase Synthesizer.
    synthesizer = Synthesizer()

    # -- Assert (Verificación) --
    # Verificamos que el constructor de TTS fue llamado una vez con el nombre del modelo correcto.
    mock_tts_class.assert_called_once_with("tts_models/es/css10/vits")
    # Verificamos que el método 'to' (para mover a CPU/GPU) fue llamado.
    mock_tts_class.return_value.to.assert_called_once()
    # Verificamos que la lista de locutores en nuestra instancia es la que definimos.
    assert synthesizer.speakers == fake_speakers
    # Verificamos que el método get_speakers() también devuelve la lista correcta.
    assert synthesizer.get_speakers() == fake_speakers

@patch('voxflow_core.TTS')
def test_synthesize_audio(mock_tts_class):
    """
    # Prueba el método de síntesis de audio.
    # Verifica que el método llama a la función tts de Coqui con los parámetros correctos
    # y devuelve los datos de audio y la frecuencia de muestreo esperados.
    """
    # -- Arrange (Preparación) --
    # Configuramos de nuevo nuestro objeto TTS simulado.
    mock_tts_instance = MagicMock()
    # Creamos un array de audio falso que simulará la salida de la síntesis.
    fake_audio_output = np.array([0.1, -0.1, 0.2], dtype=np.float32)
    # Definimos una frecuencia de muestreo falsa.
    fake_sample_rate = 24000
    # Configuramos el método 'tts' simulado para que devuelva nuestro audio falso.
    mock_tts_instance.tts.return_value = fake_audio_output
    # Configuramos la propiedad simulada para la frecuencia de muestreo.
    mock_tts_instance.synthesizer.output_sample_rate = fake_sample_rate
    mock_tts_class.return_value.to.return_value = mock_tts_instance

    # Creamos la instancia del sintetizador.
    synthesizer = Synthesizer()

    # -- Act (Acción) --
    # Llamamos al método que queremos probar.
    audio_data, sample_rate = synthesizer.synthesize("Hola mundo", "speaker_1")

    # -- Assert (Verificación) --
    # Verificamos que el método 'tts' simulado fue llamado una vez.
    mock_tts_instance.tts.assert_called_once()
    # Verificamos que fue llamado con los argumentos correctos.
    mock_tts_instance.tts.assert_called_with(text="Hola mundo", speaker="speaker_1", language="es")
    # Verificamos que los datos de audio devueltos son los que esperamos.
    np.testing.assert_array_equal(audio_data, fake_audio_output)
    # Verificamos que la frecuencia de muestreo es la correcta.
    assert sample_rate == fake_sample_rate

# Usamos la 'fixture' tmp_path de pytest, que nos proporciona una ruta a una carpeta temporal única para esta prueba.
def test_save_to_wav(tmp_path):
    """
    # Prueba la función de guardado de archivos WAV.
    # No necesita simular TTS, ya que solo maneja datos y archivos.
    # Verifica que el archivo .wav se crea y que su contenido es correcto.
    """
    # -- Arrange (Preparación) --
    # Creamos un objeto Synthesizer. Como no llamaremos a métodos que usan TTS, no necesitamos el mock aquí.
    # Para evitar el error de inicialización, podemos mockear solo el __init__.
    with patch.object(Synthesizer, '__init__', lambda x: None):
        synthesizer = Synthesizer()

    # Creamos datos de audio falsos y una frecuencia de muestreo.
    sample_rate = 22050
    audio_data = np.array([np.sin(2 * np.pi * 440 * x / sample_rate) for x in range(sample_rate)], dtype=np.float32)
    # Definimos la ruta de salida dentro de la carpeta temporal.
    output_path = os.path.join(tmp_path, "test.wav")

    # -- Act (Acción) --
    # Llamamos al método de guardado.
    result = synthesizer.save_to_wav(audio_data, sample_rate, output_path)

    # -- Assert (Verificación) --
    # Verificamos que el método devolvió True, indicando éxito.
    assert result is True
    # Verificamos que el archivo fue realmente creado en la ruta esperada.
    assert os.path.exists(output_path)

    # Leemos el archivo WAV que acabamos de crear.
    saved_rate, saved_data = read_wav(output_path)
    # Verificamos que la frecuencia de muestreo en el archivo es la correcta.
    assert saved_rate == sample_rate
    # Verificamos que los datos de audio en el archivo son idénticos a los que guardamos.
    np.testing.assert_array_almost_equal(saved_data, audio_data)
