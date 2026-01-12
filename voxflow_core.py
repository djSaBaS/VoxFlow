# Importamos los módulos necesarios.
import subprocess  # Para ejecutar procesos externos como Piper y FFmpeg.
import platform    # Para detectar el sistema operativo.
import shlex       # Para dividir cadenas de comandos en listas de forma segura.

# Definimos la clase principal que encapsula la lógica de síntesis de voz.
class Synthesizer:
    # El método constructor que se inicializa con la ruta al modelo de voz.
    def __init__(self, model_path):
        self.model_path = model_path

    # Método para reproducir texto como voz directamente.
    def say(self, text):
        # Detectamos el sistema operativo actual.
        os_type = platform.system()

        # El comando base para Piper, como una lista para evitar 'shell=True'.
        piper_cmd = ['piper', '--model', self.model_path, '--output_raw']

        # Seleccionamos el comando del reproductor de audio según el S.O.
        player_cmd_str = ""
        if os_type == "Linux":
            # 'aplay' es el estándar en muchas distribuciones de Linux.
            player_cmd_str = "aplay -r 22050 -f S16_LE -t raw"
        elif os_type == "Darwin": # 'Darwin' es el nombre del núcleo de macOS.
            # 'afplay' es el reproductor de audio nativo de macOS.
            player_cmd_str = "afplay -r 22050 -f s16le"
        elif os_type == "Windows":
            # 'ffplay' es una opción versátil para Windows (requiere FFmpeg).
            # El '-' al final indica que la entrada se recibe de la tubería (stdin).
            player_cmd_str = "ffplay -nodisp -autoexit -ar 22050 -f s16le -"
        else:
            # Si el S.O. no es compatible, lanzamos un error.
            raise NotImplementedError(f"Sistema operativo no compatible: {os_type}")

        # Convertimos la cadena del comando del reproductor en una lista segura.
        player_cmd = shlex.split(player_cmd_str)

        try:
            # Iniciamos el proceso de Piper. Su salida de audio (stdout) se redirige a una tubería.
            piper_process = subprocess.Popen(piper_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

            # Iniciamos el proceso del reproductor. Su entrada (stdin) se conecta a la salida de Piper.
            player_process = subprocess.Popen(player_cmd, stdin=piper_process.stdout)

            # Cerramos la tubería de salida de Piper en este proceso padre.
            # Esto es importante para que Piper reciba una señal SIGPIPE si el reproductor termina prematuramente.
            piper_process.stdout.close()

            # Escribimos el texto (codificado en UTF-8) a la entrada de Piper.
            piper_process.stdin.write(text.encode('utf-8'))
            # Cerramos la entrada de Piper para señalar que hemos terminado de enviar datos.
            piper_process.stdin.close()

            # Esperamos a que el proceso de Piper termine para liberar recursos.
            # No esperamos al reproductor para que la reproducción sea asíncrona y la GUI no se bloquee.
            piper_process.wait()

        except FileNotFoundError:
            # Este error ocurre si 'piper' o el comando del reproductor no se encuentran.
            print(f"Error: '{piper_cmd[0]}' o '{player_cmd[0]}' no encontrado. Asegúrate de que estén instalados y en el PATH del sistema.")
        except Exception as e:
            # Capturamos cualquier otro error inesperado durante la reproducción.
            print(f"Ocurrió un error inesperado durante la reproducción: {e}")

    # Método para guardar el texto sintetizado como un archivo MP3.
    def save_to_mp3(self, text, output_path):
        # El comando de Piper como una lista.
        piper_cmd = ['piper', '--model', self.model_path, '--output_raw']

        # El comando de FFmpeg como una lista, para convertir el audio raw a MP3.
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 's16le', '-ar', '22050', '-ac', '1', '-i', '-',
            '-acodec', 'libmp3lame', '-q:a', '2', output_path
        ]

        try:
            # Iniciamos el proceso de Piper, con su salida y error redirigidos a tuberías.
            piper_process = subprocess.Popen(piper_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            # Iniciamos FFmpeg, conectando su entrada a la salida de Piper y capturando su error estándar.
            ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=piper_process.stdout, stderr=subprocess.PIPE)

            # Cerramos la tubería de salida de Piper en el proceso padre.
            piper_process.stdout.close()

            # Pasamos el texto a Piper y cerramos su entrada.
            piper_process.stdin.write(text.encode('utf-8'))
            piper_process.stdin.close()

            # Esperamos a que FFmpeg termine y capturamos su salida de error.
            ffmpeg_stderr = ffmpeg_process.communicate()[1]

            # Esperamos a que Piper termine.
            piper_process.wait()

            # Verificamos si alguno de los procesos falló.
            if piper_process.returncode != 0:
                print(f"El proceso de Piper falló con el código de error {piper_process.returncode}")
                return False

            if ffmpeg_process.returncode != 0:
                print(f"El proceso de FFmpeg falló con el código de error {ffmpeg_process.returncode}")
                # Mostramos el error de FFmpeg, que suele ser muy útil para depurar.
                print(f"Error de FFmpeg: {ffmpeg_stderr.decode('utf-8', errors='ignore')}")
                return False

            # Si ambos procesos terminaron con éxito, retornamos True.
            return True

        except FileNotFoundError:
            # Este error ocurre si 'piper' o 'ffmpeg' no se encuentran.
            print("Error: 'piper' o 'ffmpeg' no se encontraron. Asegúrate de que estén instalados y en el PATH.")
            return False
        except Exception as e:
            # Capturamos cualquier otro error inesperado.
            print(f"Ocurrió un error inesperado al guardar el archivo MP3: {e}")
            return False
