# -*- coding: utf-8 -*-
# Importamos las bibliotecas necesarias.
import torch  # Biblioteca principal de PyTorch, necesaria para Coqui TTS.
from TTS.api import TTS  # La clase principal de la API de Coqui TTS.
import numpy as np  # NumPy, para manejar los arrays de audio.
from scipy.io.wavfile import write as write_wav  # Para guardar los datos de audio en un archivo .wav.
import os # Lo usaremos para la gestión de archivos

# Definimos la clase principal que encapsula la lógica de síntesis de voz.
class Synthesizer:
    """
    # Esta clase gestiona la inicialización del modelo TTS y la síntesis de voz.
    """
    # El método constructor de la clase.
    def __init__(self):
        """
        # Inicializa el sintetizador, cargando el modelo de Coqui TTS.
        # Detecta si hay una GPU disponible (CUDA) para un procesamiento más rápido.
        """
        # Determinamos el dispositivo a utilizar: 'cuda' si hay una GPU disponible, de lo contrario 'cpu'.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Mensaje informativo para el usuario sobre el dispositivo que se está utilizando.
        print(f"Usando dispositivo: {self.device}")

        # Ruta al modelo TTS que vamos a utilizar. Cambiamos a un modelo VITS en español que soporta selección de locutor por nombre.
        model_name = "tts_models/es/css10/vits"

        # Inicializamos la API de TTS con el modelo especificado y lo movemos al dispositivo seleccionado (GPU o CPU).
        # Esto puede tardar un momento la primera vez, ya que necesita descargar el modelo.
        self.tts = TTS(model_name).to(self.device)

        # Obtenemos y almacenamos la lista de locutores (voces) disponibles en este modelo.
        self.speakers = self.tts.speakers

    # Método para obtener la lista de voces/locutores disponibles.
    def get_speakers(self):
        """
        # Devuelve la lista de nombres de los locutores disponibles en el modelo cargado.
        #
        # Returns:
        #     list: Una lista de strings, donde cada string es el nombre de un locutor.
        """
        # Retornamos la lista de locutores que obtuvimos durante la inicialización.
        return self.speakers

    # Método principal para convertir texto a datos de audio.
    def synthesize(self, text, speaker_name):
        """
        # Sintetiza el texto dado utilizando la voz del locutor especificado.
        #
        # Args:
        #     text (str): El texto que se convertirá a voz.
        #     speaker_name (str): El nombre del locutor a utilizar para la síntesis.
        #
        # Returns:
        #     tuple: Una tupla que contiene los datos de audio como un array de NumPy
        #            y la frecuencia de muestreo (sample rate) del audio.
        """
        # Usamos el método 'tts' del objeto para generar el audio.
        # 'speaker' especifica la voz a usar y 'language' el idioma del texto.
        # La salida es una lista de floats que representa la onda de sonido.
        wav_output = self.tts.tts(text=text, speaker=speaker_name, language="es")

        # Convertimos la lista de salida a un array de NumPy con el tipo de dato float32.
        audio_data = np.array(wav_output, dtype=np.float32)

        # Obtenemos la frecuencia de muestreo del sintetizador. Es importante para la reproducción y guardado.
        sample_rate = self.tts.synthesizer.output_sample_rate

        # Añadimos una impresión de depuración para verificar el audio generado.
        print(f"Audio generado: shape={audio_data.shape}, rate={sample_rate}, min={audio_data.min()}, max={audio_data.max()}")

        # Devolvemos tanto los datos del audio como la frecuencia de muestreo.
        return audio_data, sample_rate

    # Método para guardar los datos de audio en un archivo .wav.
    def save_to_wav(self, audio_data, sample_rate, output_path):
        """
        # Guarda los datos de audio en un archivo WAV en la ruta especificada.
        #
        # Args:
        #     audio_data (np.ndarray): El array de NumPy con los datos del audio.
        #     sample_rate (int): La frecuencia de muestreo del audio.
        #     output_path (str): La ruta del archivo donde se guardará el audio.
        """
        try:
            # Usamos la función 'write_wav' de SciPy para escribir los datos en el archivo.
            # Es importante que el sample_rate sea correcto para que el audio no suene acelerado o ralentizado.
            write_wav(output_path, sample_rate, audio_data)
            # Retornamos True si el archivo se guardó con éxito.
            return True
        except Exception as e:
            # Si ocurre algún error durante el guardado, lo imprimimos en la consola.
            print(f"Error al guardar el archivo WAV en '{output_path}': {e}")
            # Retornamos False para indicar que hubo un problema.
            return False
