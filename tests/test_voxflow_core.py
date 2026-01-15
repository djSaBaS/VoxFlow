# -*- coding: utf-8 -*-
# // Importamos las bibliotecas necesarias para las pruebas.
import pytest
# // numpy para crear datos de audio falsos.
import numpy as np
# // MagicMock y patch para simular ('mockear') objetos y funciones externas, como la API de TTS o el sistema de archivos.
from unittest.mock import MagicMock, patch, call
# // os para manejar rutas de archivos de forma compatible.
import os
# // scipy.io.wavfile para leer archivos WAV y verificar su contenido en la prueba de guardado.
from scipy.io.wavfile import read as read_wav

# // Importamos la clase que vamos a probar desde el núcleo de la aplicación.
from voxflow_core import Synthesizer

# // --- Fixture de Pytest para el Sintetizador ---
# // Una fixture es una función que se ejecuta antes de cada prueba que la solicita.
# // Prepara un entorno consistente para las pruebas.
@pytest.fixture
def synthesizer():
    # // Usamos 'patch' como un gestor de contexto para simular múltiples elementos a la vez.
    with patch('voxflow_core.TTS') as mock_tts_class, \
         patch('os.path.isdir') as mock_isdir, \
         patch('os.listdir') as mock_listdir:

        # // --- Preparación del Mock de la API de TTS ---
        # // Creamos una instancia simulada de la clase TTS.
        mock_tts_instance = MagicMock()
        # // Configuramos la propiedad 'synthesizer.output_sample_rate' para que devuelva un valor fijo.
        mock_tts_instance.synthesizer.output_sample_rate = 24000
        # // Configuramos el constructor de la clase TTS simulada para que devuelva nuestra instancia simulada.
        mock_tts_class.return_value.to.return_value = mock_tts_instance

        # // --- Preparación del Mock del Sistema de Archivos ---
        # // Simulamos que el directorio 'assets/voices' existe.
        mock_isdir.return_value = True
        # // Simulamos que el directorio contiene estos tres archivos de voz.
        mock_listdir.return_value = ["male.wav", "female.wav", "child.wav"]

        # // --- Creación de la Instancia ---
        # // Creamos una instancia de nuestra clase Synthesizer. Durante su creación, usará los mocks que hemos definido.
        synth = Synthesizer()
        # // Adjuntamos el mock de la clase TTS a la instancia para poder verificarlo en las pruebas.
        synth.mock_tts_class = mock_tts_class
        # // 'yield' pasa la instancia creada a la función de prueba.
        yield synth
        # // El código después de 'yield' se ejecutaría después de la prueba (para limpieza, si fuera necesario).

# // --- Pruebas para la Clase Synthesizer ---

# // Prueba para verificar que la inicialización del Synthesizer funciona como se espera.
def test_synthesizer_initialization(synthesizer):
    # // Verifica que la clase Synthesizer se inicializa correctamente.
    # // Comprueba que el modelo TTS correcto se carga y que las voces de referencia se descubren y mapean adecuadamente.

    # // -- Verificación --
    # // Verificamos que el constructor de TTS fue llamado una vez con el nombre del modelo XTTSv2.
    synthesizer.mock_tts_class.assert_called_once_with("tts_models/multilingual/multi-dataset/xtts_v2")
    # // Verificamos que el método .to() fue llamado en la instancia retornada por el constructor.
    synthesizer.mock_tts_class.return_value.to.assert_called_once()

    # // Verificamos que los nombres amigables de las voces se han creado correctamente.
    # // El orden debe ser el correcto porque simulamos 'sorted(os.listdir(...))'.
    expected_speakers = ["Voz Infantil", "Voz Femenina", "Voz Masculina"]
    assert synthesizer.get_speakers() == expected_speakers

    # // Verificamos que las rutas a los archivos de voz son las correctas.
    assert synthesizer.voice_references["Voz Masculina"] == os.path.join("assets", "voices", "male.wav")

# // Prueba para el método de síntesis de audio.
def test_synthesize_audio(synthesizer):
    # // Prueba que el método de síntesis llama a la función de Coqui TTS con los parámetros correctos.
    # // Verifica que se usa 'speaker_wav' y 'language', y que devuelve los datos esperados.

    # // -- Preparación --
    # // Creamos un array de audio falso que simulará la salida de la síntesis.
    fake_audio_output = np.array([0.1, -0.1, 0.2], dtype=np.float32)
    # // Configuramos el método 'tts' simulado para que devuelva nuestro audio falso.
    synthesizer.tts.tts.return_value = fake_audio_output

    # // -- Acción --
    # // Llamamos al método que queremos probar con un texto y una voz de ejemplo.
    audio_data, sample_rate = synthesizer.synthesize("Hola mundo", "Voz Femenina")

    # // -- Verificación --
    # // Verificamos que el método 'tts' simulado fue llamado una vez.
    synthesizer.tts.tts.assert_called_once()
    # // Verificamos que fue llamado con los argumentos correctos.
    synthesizer.tts.tts.assert_called_with(
        text="Hola mundo",
        speaker_wav=os.path.join("assets", "voices", "female.wav"),
        language="es"
    )
    # // Comparamos el array de audio devuelto con el que esperamos.
    np.testing.assert_array_equal(audio_data, fake_audio_output)
    # // Verificamos que la frecuencia de muestreo es la correcta.
    assert sample_rate == 24000

# // Prueba para la función de guardado de archivos WAV.
def test_save_to_wav(tmp_path):
    # // Prueba que el guardado de archivos .wav funciona correctamente.
    # // No necesita mocks de TTS, ya que solo maneja datos y archivos.

    # // -- Preparación --
    # // Como esta prueba no usa la fixture, creamos una instancia 'vacía' de Synthesizer.
    # // Usamos 'patch' para saltarnos el __init__ y evitar la carga real del modelo.
    with patch.object(Synthesizer, '__init__', lambda x: None):
        synthesizer = Synthesizer()

    # // Creamos datos de audio falsos y una frecuencia de muestreo.
    sample_rate = 22050
    audio_data = np.random.randn(sample_rate).astype(np.float32)
    # // Definimos la ruta de salida dentro de la carpeta temporal que nos proporciona pytest.
    output_path = tmp_path / "test.wav"

    # // -- Acción --
    # // Llamamos al método de guardado.
    result = synthesizer.save_to_wav(audio_data, sample_rate, str(output_path))

    # // -- Verificación --
    # // Verificamos que el método devolvió True, indicando éxito.
    assert result is True
    # // Verificamos que el archivo fue realmente creado en la ruta esperada.
    assert output_path.exists()

    # // Leemos el archivo WAV que acabamos de crear para comprobar su contenido.
    saved_rate, saved_data = read_wav(output_path)
    # // Verificamos que la frecuencia de muestreo en el archivo es la correcta.
    assert saved_rate == sample_rate
    # // Verificamos que los datos de audio en el archivo son idénticos a los que guardamos.
    np.testing.assert_array_almost_equal(saved_data, audio_data)
