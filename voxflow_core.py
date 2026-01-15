# -*- coding: utf-8 -*-
# // Importamos las bibliotecas necesarias.
# // torch: Biblioteca principal de PyTorch, fundamental para el funcionamiento de Coqui TTS.
import torch
# // TTS: La clase principal de la API de Coqui TTS, que nos da acceso a los modelos preentrenados.
from TTS.api import TTS
# // numpy: Imprescindible para el manejo de los arrays de audio, que se representan como arrays numéricos.
import numpy as np
# // write_wav: Función específica de SciPy para guardar arrays de NumPy en formato de archivo .wav.
from scipy.io.wavfile import write as write_wav
# // os: Módulo del sistema operativo, necesario para la gestión de rutas y archivos (ej: crear carpetas, unir rutas).
import os

# // Definimos la clase principal que encapsula toda la lógica de la síntesis de voz.
class Synthesizer:
    # // Esta clase se encarga de todo: desde cargar el modelo de IA hasta generar y guardar el audio.
    # // El método constructor de la clase, que se ejecuta automáticamente al crear un objeto Synthesizer.
    def __init__(self):
        # // Inicializa el sintetizador cargando el modelo de Coqui TTS y preparando el entorno.
        # // También detecta si hay una GPU disponible (CUDA) para acelerar el procesamiento.
        # // Determinamos el dispositivo a utilizar: 'cuda' si hay una GPU compatible, de lo contrario, 'cpu'.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # // Mensaje informativo para el usuario, mostrando qué dispositivo se está utilizando (GPU es más rápida).
        print(f"Usando dispositivo: {self.device}")

        # // Ruta al modelo TTS que vamos a utilizar.
        # // Usaremos 'xtts_v2', un modelo de alta calidad que es multilingüe y soporta clonación de voz.
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

        # // Inicializamos la API de TTS con el modelo especificado y lo movemos al dispositivo seleccionado (GPU o CPU).
        # // La primera vez que se ejecute, este paso puede tardar, ya que necesita descargar los archivos del modelo.
        self.tts = TTS(model_name).to(self.device)

        # // Diccionario para almacenar las rutas a nuestras voces de referencia.
        # // La clave será un nombre amigable para el usuario (ej: "Voz Femenina") y el valor será la ruta al archivo .wav.
        self.voice_references = {}
        # // Llamamos al método interno para que cargue las voces desde el directorio 'assets/voices'.
        self._load_voice_references()

    # // Método privado para cargar las voces de referencia.
    def _load_voice_references(self):
        # // Este método escanea el directorio 'assets/voices', identifica los archivos .wav
        # // y los mapea a nombres descriptivos que se mostrarán en la interfaz de usuario.
        # // Define la ruta relativa al directorio que contiene las voces de referencia.
        voices_dir = "assets/voices"
        # // Comprobamos si el directorio existe para evitar errores en caso de que se haya borrado o no se haya creado.
        if not os.path.isdir(voices_dir):
            # // Si el directorio no se encuentra, se imprime una advertencia en la consola.
            print(f"Advertencia: El directorio de voces '{voices_dir}' no se encuentra.")
            # // Salimos del método para evitar que el código siga ejecutándose sin las voces.
            return

        # // Iteramos sobre cada archivo en el directorio de voces, ordenado alfabéticamente para mantener la consistencia.
        for filename in sorted(os.listdir(voices_dir)):
            # // Nos aseguramos de procesar únicamente los archivos que terminan en .wav para ignorar otros posibles archivos.
            if filename.endswith(".wav"):
                # // Extraemos el nombre del archivo sin la extensión (ej: "male" de "male.wav").
                name = os.path.splitext(filename)[0]
                # // Asignamos nombres más descriptivos y amigables basados en el nombre del archivo.
                if name == 'child':
                    # // Si el archivo se llama 'child.wav', en la app se mostrará como 'Voz Infantil'.
                    pretty_name = "Voz Infantil"
                elif name == 'female':
                    # // Si es 'female.wav', se mostrará como 'Voz Femenina'.
                    pretty_name = "Voz Femenina"
                elif name == 'male':
                    # // Si es 'male.wav', se mostrará como 'Voz Masculina'.
                    pretty_name = "Voz Masculina"
                else:
                    # // Si el archivo tiene otro nombre, lo formateamos automáticamente (ej: "custom" -> "Voz Custom").
                    pretty_name = f"Voz {name.capitalize()}"

                # // Construimos la ruta completa y compatible con cualquier sistema operativo al archivo de audio.
                file_path = os.path.join(voices_dir, filename)
                # // Almacenamos en nuestro diccionario la correspondencia entre el nombre amigable y la ruta del archivo.
                self.voice_references[pretty_name] = file_path

        # // Informamos al usuario a través de la consola qué voces se han cargado exitosamente.
        print(f"Voces de referencia cargadas: {list(self.voice_references.keys())}")

    # // Método público para obtener la lista de voces disponibles.
    def get_speakers(self):
        # // Devuelve una lista con los nombres amigables de las voces de referencia que hemos cargado.
        # // Esto se usará para poblar el menú desplegable en la interfaz de usuario.
        # // Returns:
        # //     list: Una lista de strings, donde cada string es un nombre de voz (ej: ["Voz Femenina", "Voz Masculina"]).
        # // Devolvemos las "claves" de nuestro diccionario, que son los nombres amigables que hemos definido.
        return list(self.voice_references.keys())

    # // Método principal para convertir un texto a datos de audio.
    def synthesize(self, text, speaker_name):
        # // Sintetiza el texto proporcionado utilizando la voz de referencia que el usuario ha seleccionado.
        # // Args:
        # //     text (str): El texto que se va a convertir a voz.
        # //     speaker_name (str): El nombre amigable de la voz de referencia a utilizar (ej: "Voz Masculina").
        # // Returns:
        # //     tuple: Una tupla con (datos_de_audio, frecuencia_de_muestreo), o (None, None) si ocurre un error.
        # // Buscamos en nuestro diccionario la ruta del archivo .wav correspondiente al nombre amigable de la voz.
        speaker_wav_path = self.voice_references.get(speaker_name)

        # // Si no se encuentra una ruta para la voz seleccionada (lo que sería un error inesperado), lo gestionamos.
        if not speaker_wav_path:
            # // Informamos del problema en la consola para facilitar la depuración.
            print(f"Error: No se encontró la ruta del archivo de voz para '{speaker_name}'")
            # // Devolvemos None para que la interfaz de usuario sepa que la síntesis ha fallado.
            return None, None

        # // Imprimimos la ruta del archivo de clonación que se va a usar, útil para depuración.
        print(f"Usando archivo de voz de referencia: {speaker_wav_path}")

        # // Llamamos al método 'tts' del modelo XTTSv2, que es el que hace la "magia".
        # // - text: El texto a sintetizar.
        # // - speaker_wav: La ruta al archivo de audio que usará como referencia para clonar la voz.
        # // - language: El código del idioma del texto (ej: "es" para español). Es crucial para este modelo multilingüe.
        wav_output = self.tts.tts(
            text=text,
            speaker_wav=speaker_wav_path,
            language="es"
        )

        # // Convertimos la lista de salida del modelo a un array de NumPy con el tipo de dato float32, estándar para audio.
        audio_data = np.array(wav_output, dtype=np.float32)

        # // Obtenemos la frecuencia de muestreo (sample rate) del sintetizador. Es vital para la reproducción y guardado correctos.
        sample_rate = self.tts.synthesizer.output_sample_rate

        # // Añadimos una impresión de depuración para verificar las propiedades del audio generado.
        print(f"Audio generado: shape={audio_data.shape}, rate={sample_rate}, min={audio_data.min()}, max={audio_data.max()}")

        # // Devolvemos tanto los datos del audio como la frecuencia de muestreo para que puedan ser utilizados por otros métodos.
        return audio_data, sample_rate

    # // Método para guardar los datos de audio en un archivo .wav.
    def save_to_wav(self, audio_data, sample_rate, output_path):
        # // Guarda los datos de audio en un archivo WAV en la ruta que el usuario ha especificado.
        # // Args:
        # //     audio_data (np.ndarray): El array de NumPy con los datos del audio a guardar.
        # //     sample_rate (int): La frecuencia de muestreo del audio.
        # //     output_path (str): La ruta completa del archivo donde se guardará el audio.
        try:
            # // Usamos la función 'write_wav' de SciPy para escribir los datos numéricos en un archivo .wav.
            # // Es muy importante que la frecuencia de muestreo sea la correcta para que el audio no suene distorsionado.
            write_wav(output_path, sample_rate, audio_data)
            # // Si el archivo se guarda sin errores, devolvemos True para indicar que la operación fue exitosa.
            return True
        except Exception as e:
            # // Si ocurre cualquier error durante el guardado (ej: permisos de escritura), lo capturamos.
            print(f"Error al guardar el archivo WAV en '{output_path}': {e}")
            # // Devolvemos False para que la aplicación sepa que el guardado ha fallado.
            return False
