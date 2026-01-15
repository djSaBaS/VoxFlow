# -*- coding: utf-8 -*-
# // Importamos las librerías necesarias para la aplicación.
import customtkinter as ctk                     # // Librería para construir la interfaz gráfica moderna.
from customtkinter import filedialog            # // Módulo específico para los diálogos de abrir/guardar archivos.
import threading                                # // Para ejecutar procesos pesados (como la síntesis) en hilos separados y no bloquear la UI.
import sounddevice as sd                        # // Para reproducir y detener el audio directamente desde la aplicación.
import os                                       # // Para interactuar con el sistema operativo, principalmente para manejar rutas de archivos.
import json                                     # // Para guardar y cargar la configuración de la aplicación en formato JSON.
import matplotlib.pyplot as plt                 # // Para generar el gráfico de la forma de onda.
from PIL import Image                           # // Para manejar la imagen del gráfico y convertirla a un formato que CustomTkinter pueda usar.
import io                                       # // Para manejar el gráfico como un flujo de bytes en memoria, evitando archivos temporales.
import numpy as np                              # // Para el análisis numérico de los datos de audio.

# // Importamos la lógica principal de síntesis de voz desde nuestro módulo 'voxflow_core'.
from voxflow_core import Synthesizer

# // --- CONFIGURACIÓN GLOBAL DE LA APARIENCIA ---
# // Establecemos el modo de apariencia por defecto (Light, Dark, System).
ctk.set_appearance_mode("light") 
# // Establecemos el tema de color por defecto para los widgets.
ctk.set_default_color_theme("blue")

# // Definimos la clase principal de nuestra aplicación, que hereda de ctk.CTk (la ventana principal).
class VoxFlowApp(ctk.CTk):
    # // El método constructor, que se llama al crear una nueva instancia de la aplicación.
    def __init__(self):
        # // Llamamos al constructor de la clase padre (CTk) para inicializar la ventana.
        super().__init__()

        # // --- 1. CONFIGURACIÓN DE LA VENTANA PRINCIPAL ---
        # // Establecemos el título que aparecerá en la barra de la ventana.
        self.title("VoxFlow 2.0 - Coqui TTS")
        # // Definimos el tamaño inicial de la ventana en píxeles (ancho x alto).
        self.geometry("800x600")
        # // Configuramos el sistema de rejilla (grid) para que la ventana sea redimensionable.
        # // La columna 0 se expandirá horizontalmente para ocupar el espacio extra.
        self.grid_columnconfigure(0, weight=1)
        # // La fila 3 (donde está el cuadro de texto) se expandirá verticalmente.
        self.grid_rowconfigure(3, weight=1)

        # // --- 2. GESTIÓN DEL ESTADO INTERNO ---
        # // Variable para almacenar la instancia de nuestra clase Synthesizer. Se inicializa más tarde en un hilo.
        self.synthesizer = None
        # // Buffer para guardar los datos del audio generado como un array de NumPy.
        self.audio_data = None
        # // Variable para almacenar la frecuencia de muestreo del audio, necesaria para la reproducción y guardado.
        self.sample_rate = None
        # // Variable para mantener una referencia a la imagen de la onda y evitar que Python la elimine prematuramente.
        self.waveform_image = None

        # // --- 3. CREACIÓN Y POSICIONAMIENTO DE WIDGETS ---

        # // --- 3.1. Barra de Menú Superior ---
        # // Creamos un frame (contenedor) para agrupar los controles del menú.
        self.menu_bar = ctk.CTkFrame(self)
        # // Lo posicionamos en la primera fila (0) y columna (0), haciendo que se expanda a lo ancho.
        self.menu_bar.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        # // Hacemos que la columna 2 del menú se expanda para empujar el selector de voz hacia la derecha.
        self.menu_bar.grid_columnconfigure(2, weight=1)

        # // Botón para cargar una configuración guardada. Al pulsarlo, llama al método 'load_configuration'.
        self.btn_load_config = ctk.CTkButton(self.menu_bar, text="Cargar Config", command=self.load_configuration)
        self.btn_load_config.grid(row=0, column=0, padx=5, pady=5)
        # // Botón para guardar la configuración actual. Llama al método 'save_configuration'.
        self.btn_save_config = ctk.CTkButton(self.menu_bar, text="Guardar Config", command=self.save_configuration)
        self.btn_save_config.grid(row=0, column=1, padx=5, pady=5)

        # // Menú desplegable para que el usuario elija la voz de referencia.
        self.speaker_dropdown = ctk.CTkComboBox(self.menu_bar, values=["Cargando..."], width=200)
        # // Lo alineamos a la derecha (este) dentro de su celda.
        self.speaker_dropdown.grid(row=0, column=3, padx=10, pady=10, sticky="e")
        # // Establecemos el texto que se muestra por defecto antes de que las voces se carguen.
        self.speaker_dropdown.set("Selecciona Voz")

        # // --- 3.2. Etiqueta de Estado ---
        # // Etiqueta para mostrar mensajes informativos al usuario. La separamos del menú para mayor claridad.
        self.status_label = ctk.CTkLabel(self, text="Iniciando IA, por favor espera...", font=("Arial", 14))
        # // La posicionamos en la segunda fila, alineada a la izquierda (oeste).
        self.status_label.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="w")

        # // --- 3.3. Visualizador de Onda ---
        # // Usamos un Label como lienzo para mostrar la imagen del gráfico de la forma de onda.
        self.waveform_label = ctk.CTkLabel(self, text="", height=100)
        self.waveform_label.grid(row=2, column=0, columnspan=2, padx=20, pady=5, sticky="ew")

        # // --- 3.4. Campo de Texto Principal ---
        # // Área de texto multilínea donde el usuario introducirá el texto a sintetizar.
        self.text_input = ctk.CTkTextbox(self, font=("Arial", 16))
        # // Lo posicionamos para que ocupe todo el espacio central expandible.
        self.text_input.grid(row=3, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")
        # // Insertamos un texto de ejemplo para guiar al usuario sobre qué hacer.
        self.text_input.insert("0.0", "Escribe aquí el texto que deseas convertir a voz...")

        # // --- 3.5. Contenedor de Botones de Control ---
        # // Creamos otro frame para agrupar los botones de acción principales.
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.grid(row=4, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        # // Hacemos que las 4 columnas del frame tengan el mismo peso para que los botones se distribuyan uniformemente.
        self.button_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # // --- 3.6. Botones de Acción ---
        # // Botón para iniciar la síntesis de voz. Llama al método 'start_synthesis'.
        self.btn_generate = ctk.CTkButton(self.button_frame, text="Generar Audio", command=self.start_synthesis)
        self.btn_generate.grid(row=0, column=0, padx=10, pady=10)
        # // Lo desactivamos por defecto; se activará solo cuando el modelo de IA esté listo.
        self.btn_generate.configure(state="disabled")

        # // Botón para reproducir el audio generado. Llama a 'play_audio'. Color verde para una acción positiva.
        self.btn_play = ctk.CTkButton(self.button_frame, text="Reproducir", command=self.play_audio, fg_color="green")
        self.btn_play.grid(row=0, column=1, padx=10, pady=10)
        # // También desactivado por defecto. Se activa solo después de generar un audio.
        self.btn_play.configure(state="disabled")

        # // Botón para detener la reproducción. Usa una lambda para llamar directamente a 'sd.stop()'. Color rojo para acción de parada.
        self.btn_stop = ctk.CTkButton(self.button_frame, text="Detener", command=lambda: sd.stop(), fg_color="red")
        self.btn_stop.grid(row=0, column=2, padx=10, pady=10)

        # // Botón para guardar el audio en un archivo. Llama al método 'save_audio'.
        self.btn_save = ctk.CTkButton(self.button_frame, text="Guardar Audio", command=self.save_audio)
        self.btn_save.grid(row=0, column=3, padx=10, pady=10)
        # // Desactivado hasta que haya un audio disponible para guardar.
        self.btn_save.configure(state="disabled")

        # // --- 4. INICIALIZACIÓN ASÍNCRONA DEL MOTOR ---
        # // Creamos e iniciamos un hilo para cargar el motor de IA. Esto evita que la ventana se congele durante la carga.
        # // 'daemon=True' asegura que este hilo se cierre automáticamente cuando se cierra la aplicación.
        threading.Thread(target=self.load_engine, daemon=True).start()

    # // Método para cargar el motor de IA en un hilo secundario.
    def load_engine(self):
        # // Carga el sintetizador de Coqui TTS y prepara la interfaz de usuario.
        try:
            # // Creamos la instancia de nuestra clase Synthesizer. Este paso puede tardar.
            self.synthesizer = Synthesizer()
            # // Una vez cargado, programamos las actualizaciones de la UI para que se ejecuten en el hilo principal.
            # // self.after(0, ...) le dice al bucle principal de la app que ejecute la función tan pronto como sea posible.
            self.after(0, self._on_engine_loaded)
        except Exception as e:
            # // Si ocurre un error, también lo comunicamos de forma segura en el hilo principal.
            self.after(0, lambda: self.status_label.configure(text=f"Error al cargar: {e}"))

    # // Método que se ejecuta en el hilo principal para actualizar la UI una vez que el motor está listo.
    def _on_engine_loaded(self):
        # // Obtenemos la lista de nombres de voces amigables desde el sintetizador.
        speakers = self.synthesizer.get_speakers()
        # // Si se encontraron voces, las configuramos en el menú desplegable.
        if speakers:
            # // 'configure' actualiza la lista de opciones del menú desplegable.
            self.speaker_dropdown.configure(values=speakers)
            # // Establecemos la primera voz de la lista como la opción seleccionada por defecto.
            self.speaker_dropdown.set(speakers[0])
            # // Actualizamos la UI para reflejar que todo está listo.
            self.status_label.configure(text="Motor IA listo para usar")
            # // Habilitamos el botón de "Generar Audio".
            self.btn_generate.configure(state="normal")
        else:
            # // Si no se encontraron voces, mostramos un mensaje de error.
            self.status_label.configure(text="Error: No se encontraron voces de referencia.")
            self.speaker_dropdown.configure(values=["Sin voces"])
            self.speaker_dropdown.set("Sin voces")

    # // Método que se llama al pulsar el botón "Generar Audio".
    def start_synthesis(self):
        # // Inicia el proceso de síntesis de voz en un nuevo hilo para no bloquear la interfaz.
        # // Obtenemos todo el texto del cuadro de texto, desde el principio ("1.0") hasta el final ("end-1c").
        text = self.text_input.get("1.0", "end-1c")
        # // Obtenemos la voz que el usuario ha seleccionado en el menú desplegable.
        voice = self.speaker_dropdown.get()
        
        # // Validamos que el usuario haya escrito algo (ignorando espacios en blanco).
        if not text.strip():
            self.status_label.configure(text="Por favor, escribe algo de texto.")
            return

        # // Deshabilitamos el botón de generar para evitar clics múltiples mientras se procesa.
        self.btn_generate.configure(state="disabled")
        # // Informamos al usuario que el proceso ha comenzado.
        self.status_label.configure(text="Generando audio, por favor espera...")
        
        # // Creamos e iniciamos el hilo que ejecutará el proceso pesado de síntesis.
        threading.Thread(target=self.run_process, args=(text, voice), daemon=True).start()

    # // Método que se ejecuta en el hilo secundario para realizar la síntesis.
    def run_process(self, text, voice):
        # // Llama al sintetizador y maneja el resultado.
        try:
            # // Llamamos al método 'synthesize' de nuestra clase Synthesizer.
            self.audio_data, self.sample_rate = self.synthesizer.synthesize(text, voice)
            # // Una vez completada la síntesis, programamos la actualización de la UI en el hilo principal.
            self.after(0, self._on_synthesis_complete)
        except Exception as e:
            # // Si hay un error, lo mostramos de forma segura.
            self.after(0, lambda: self.status_label.configure(text=f"Error inesperado: {e}"))
        finally:
            # // Siempre volvemos a habilitar el botón de generar, también de forma segura.
            self.after(0, lambda: self.btn_generate.configure(state="normal"))

    # // Método que se ejecuta en el hilo principal para actualizar la UI después de la síntesis.
    def _on_synthesis_complete(self):
        # // Si la síntesis fue exitosa y tenemos datos de audio.
        if self.audio_data is not None:
            # // Actualizamos el mensaje de estado para informar al usuario.
            self.status_label.configure(text="Audio generado. Listo para reproducir o guardar.")
            # // Habilitamos los botones de reproducir y guardar.
            self.btn_play.configure(state="normal")
            self.btn_save.configure(state="normal")
            # // Llamamos al método para generar y mostrar la nueva forma de onda.
            self.update_waveform()
        else:
            # // Si la síntesis falló (devolvió None), lo comunicamos.
            self.status_label.configure(text="Error: La síntesis de voz no pudo generar audio.")

    # // Método para reproducir el audio.
    def play_audio(self):
        # // Reproduce el audio almacenado en memoria.
        # // Solo intenta reproducir si 'audio_data' no es None.
        if self.audio_data is not None:
            # // Usa la librería sounddevice para la reproducción.
            sd.play(self.audio_data, self.sample_rate)

    # // Método para guardar el audio en un archivo.
    def save_audio(self):
        # // Abre un diálogo para guardar el audio generado en un archivo .wav.
        # // Verificación de seguridad por si se llama a este método sin audio disponible.
        if self.audio_data is None:
            self.status_label.configure(text="No hay audio para guardar.")
            return

        # // Abre el diálogo de "Guardar como..." del sistema operativo.
        file_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("Archivos WAV", "*.wav")]
        )

        # // Si el usuario cierra el diálogo, 'file_path' estará vacío, por lo que no hacemos nada.
        if not file_path:
            self.status_label.configure(text="Guardado cancelado.")
            return

        # // Llamamos al método de guardado de nuestro sintetizador.
        success = self.synthesizer.save_to_wav(self.audio_data, self.sample_rate, file_path)

        # // Informamos al usuario del resultado de la operación.
        if success:
            self.status_label.configure(text=f"Audio guardado en: {os.path.basename(file_path)}")
        else:
            self.status_label.configure(text="Error al guardar el archivo de audio.")

    # // Método para generar y mostrar la visualización de la forma de onda.
    def update_waveform(self):
        # // Si no hay datos de audio, no hace nada.
        if self.audio_data is None:
            return

        # // --- 1. Generación del Gráfico con Matplotlib ---
        # // Crea una figura y un eje para el gráfico. 'figsize' controla el tamaño.
        fig, ax = plt.subplots(figsize=(8, 1.5))
        # // Crea un array de tiempo para el eje X del gráfico.
        time = np.linspace(0., len(self.audio_data) / self.sample_rate, num=len(self.audio_data))
        # // Dibuja la línea de la forma de onda. 'lw' es el grosor de la línea.
        ax.plot(time, self.audio_data, lw=1)
        # // Oculta los ejes X e Y para un diseño más limpio.
        ax.axis('off')
        # // Ajusta el diseño para eliminar cualquier borde blanco.
        fig.tight_layout(pad=0)

        # // --- 2. Conversión del Gráfico a Imagen en Memoria ---
        # // Crea un buffer de bytes en memoria para no tener que guardar un archivo temporal.
        buf = io.BytesIO()
        # // Guarda la figura en el buffer en formato PNG.
        fig.savefig(buf, format='png', dpi=100)
        # // Cierra la figura para liberar la memoria de Matplotlib.
        plt.close(fig)
        # // Regresa al inicio del buffer para que pueda ser leído.
        buf.seek(0)

        # // --- 3. Carga y Visualización de la Imagen en CustomTkinter ---
        # // Abre los datos de la imagen desde el buffer usando Pillow.
        image = Image.open(buf)
        # // Crea un objeto CTkImage, que es el formato que CustomTkinter necesita.
        self.waveform_image = ctk.CTkImage(light_image=image, dark_image=image, size=(800, 100))
        # // Configura el label del visualizador para que muestre la imagen.
        self.waveform_label.configure(image=self.waveform_image)

    # // Método para guardar la configuración actual.
    def save_configuration(self):
        # // Guarda el texto y la voz seleccionada en un archivo JSON.
        config = {
            "text": self.text_input.get("1.0", "end-1c"),
            "speaker": self.speaker_dropdown.get()
        }
        # // Abre el diálogo de "Guardar como..." para archivos JSON.
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("Archivos JSON", "*.json")]
        )
        # // Si el usuario selecciona una ruta, guarda el archivo.
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            self.status_label.configure(text=f"Configuración guardada en {os.path.basename(file_path)}")

    # // Método para cargar una configuración.
    def load_configuration(self):
        # // Carga la configuración desde un archivo JSON y actualiza la UI.
        # // Abre el diálogo de "Abrir..." para archivos JSON.
        file_path = filedialog.askopenfilename(
            filetypes=[("Archivos JSON", "*.json")]
        )
        # // Si el usuario selecciona un archivo, lo carga.
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # // Borra el texto actual e inserta el texto cargado.
            self.text_input.delete("1.0", "end")
            self.text_input.insert("1.0", config.get("text", ""))

            # // Comprueba si la voz guardada está disponible en el menú desplegable antes de seleccionarla.
            if config.get("speaker") in self.speaker_dropdown.cget("values"):
                self.speaker_dropdown.set(config.get("speaker"))

            self.status_label.configure(text=f"Configuración cargada desde {os.path.basename(file_path)}")

# // --- PUNTO DE ENTRADA DE LA APLICACIÓN ---
# // Este bloque de código se ejecuta solo si el script es ejecutado directamente.
if __name__ == "__main__":
    # // Creamos una instancia de nuestra clase de aplicación.
    app = VoxFlowApp()
    # // Iniciamos el bucle principal de la aplicación, que la mantiene corriendo y escuchando eventos.
    app.mainloop()
