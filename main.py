# -*- coding: utf-8 -*-
import customtkinter as ctk     # Librería para la interfaz moderna
import threading                # Para no bloquear la ventana durante el proceso
import sounddevice as sd        # Para reproducir el audio
import os                       # Para manejo de rutas

# Importamos tu lógica del sintetizador
from voxflow_core import Synthesizer

# Configuración del tema visual
ctk.set_appearance_mode("light") 
ctk.set_default_color_theme("blue")

class VoxFlowApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- 1. CONFIGURACIÓN DE LA VENTANA ---
        self.title("VoxFlow 2.0 - Coqui TTS")  # Título de la app
        self.geometry("800x600")               # Tamaño inicial
        self.grid_columnconfigure(0, weight=1) # Columna principal expandible
        self.grid_rowconfigure(2, weight=1)    # Fila del texto expandible

        # --- 2. GESTIÓN DE ESTADO ---
        self.synthesizer = None                # Instancia de la IA
        self.audio_data = None                 # Buffer de audio
        self.sample_rate = None                # Frecuencia de muestreo

        # --- 3. CREACIÓN DE COMPONENTES ---

        # Etiqueta de estado
        self.status_label = ctk.CTkLabel(self, text="Iniciando IA, por favor espera...", font=("Arial", 14))
        self.status_label.grid(row=0, column=0, padx=20, pady=10)

        # Selector de voz (Dropdown)
        self.speaker_dropdown = ctk.CTkComboBox(self, values=["Cargando..."], width=250)
        self.speaker_dropdown.grid(row=1, column=0, padx=20, pady=10)
        self.speaker_dropdown.set("Selecciona Voz")

        # Campo de texto multilínea
        self.text_input = ctk.CTkTextbox(self, font=("Arial", 14))
        self.text_input.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")

        # Contenedor de botones de control
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.grid(row=3, column=0, padx=20, pady=20, sticky="ew")
        self.button_frame.grid_columnconfigure((0, 1, 2), weight=1)

        # Botón para generar el audio
        self.btn_generate = ctk.CTkButton(self.button_frame, text="Generar Audio", command=self.start_synthesis)
        self.btn_generate.grid(row=0, column=0, padx=10, pady=10)
        self.btn_generate.configure(state="disabled") # Desactivado hasta que cargue la IA

        # Botón para reproducir
        self.btn_play = ctk.CTkButton(self.button_frame, text="Reproducir", command=self.play_audio, fg_color="green")
        self.btn_play.grid(row=0, column=1, padx=10, pady=10)
        self.btn_play.configure(state="disabled")

        # Botón para detener
        self.btn_stop = ctk.CTkButton(self.button_frame, text="Detener", command=lambda: sd.stop(), fg_color="red")
        self.btn_stop.grid(row=0, column=2, padx=10, pady=10)

        # --- 4. INICIALIZACIÓN ASÍNCRONA ---
        threading.Thread(target=self.load_engine, daemon=True).start()

    def load_engine(self):
        """Carga el motor de IA sin congelar la ventana."""
        try:
            self.synthesizer = Synthesizer()               # Inicializa Coqui TTS
            speakers = self.synthesizer.get_speakers()    # Obtiene lista de voces
            self.speaker_dropdown.configure(values=speakers)
            if speakers:
                self.speaker_dropdown.set(speakers[0])
            
            # Actualiza la UI una vez listo
            self.status_label.configure(text="Motor IA listo para usar")
            self.btn_generate.configure(state="normal")
        except Exception as e:
            self.status_label.configure(text=f"Error al cargar: {e}")

    def start_synthesis(self):
        """Lanza la síntesis en un hilo nuevo para no bloquear la app."""
        text = self.text_input.get("1.0", "end-1c") # Obtiene el texto del cuadro
        voice = self.speaker_dropdown.get()         # Obtiene la voz seleccionada
        
        if not text.strip():
            self.status_label.configure(text="Por favor, escribe algo")
            return

        self.btn_generate.configure(state="disabled")
        self.status_label.configure(text="Generando audio...")
        
        # Ejecuta el proceso pesado fuera del hilo principal
        threading.Thread(target=self.run_process, args=(text, voice), daemon=True).start()

    def run_process(self, text, voice):
        """Proceso interno de síntesis de audio."""
        try:
            # Llama a tu función de síntesis en voxflow_core
            self.audio_data, self.sample_rate = self.synthesizer.synthesize(text, voice)
            
            # Habilita los controles al terminar
            self.status_label.configure(text="Audio generado con éxito")
            self.btn_play.configure(state="normal")
            self.btn_generate.configure(state="normal")
        except Exception as e:
            self.status_label.configure(text=f"Error: {e}")
            self.btn_generate.configure(state="normal")

    def play_audio(self):
        """Reproduce el audio almacenado en memoria."""
        if self.audio_data is not None:
            sd.play(self.audio_data, self.sample_rate)

# --- 5. ARRANQUE DE LA APLICACIÓN ---
if __name__ == "__main__":
    app = VoxFlowApp()
    app.mainloop() # Inicia el ciclo de vida de la ventana
