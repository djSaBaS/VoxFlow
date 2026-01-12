# -*- coding: utf-8 -*-
# Importamos las bibliotecas necesarias.
import flet as ft  # El framework para la interfaz gráfica.
import threading  # Para ejecutar tareas largas (como la síntesis) en segundo plano y no bloquear la UI.
import sounddevice as sd  # Para reproducir el audio directamente.
import numpy as np  # Requerido para manejar los datos de audio.
import matplotlib.pyplot as plt  # Para crear el gráfico de la onda de sonido.
import time  # Para pequeñas pausas y control de la UI.
import json # Para guardar y cargar las configuraciones de voz.
import os # Para crear el directorio de assets.

# Importamos nuestra clase Synthesizer del otro archivo.
from voxflow_core import Synthesizer

# La función principal que define y ejecuta la aplicación de Flet.
def main(page: ft.Page):
    # --- 1. CONFIGURACIÓN INICIAL DE LA VENTANA ---

    # Título de la ventana de la aplicación.
    page.title = "VoxFlow 2.0 - con Coqui TTS"
    # Establecemos un tema claro.
    page.theme_mode = ft.ThemeMode.LIGHT
    # Añadimos un poco de espacio alrededor de los bordes.
    page.padding = 20
    # Definimos un tamaño mínimo para la ventana.
    page.window_min_width = 800
    page.window_min_height = 700

    # --- 2. ESTADO DE LA APLICACIÓN ---

    # Variable global para mantener la instancia del sintetizador.
    synthesizer = None
    # Diccionario para almacenar los datos de audio y su frecuencia de muestreo.
    audio_cache = {"data": None, "rate": None}
    # Variable para controlar si el audio se está reproduciendo.
    is_playing = threading.Event()
    # Diccionario para almacenar las configuraciones de voz guardadas por el usuario.
    saved_voice_configs = {}

    # --- GESTOR DE ESTADO PARA DESHACER/REHACER ---
    class StateManager:
        def __init__(self):
            # El historial almacena los estados pasados.
            self.history = []
            # El futuro almacena los estados deshechos.
            self.future = []

        def add_state(self, state):
            # Añadimos el estado actual al historial.
            self.history.append(state)
            # Cada vez que se añade un nuevo estado, el futuro se borra.
            self.future.clear()

        def undo(self):
            # Si hay algo en el historial para deshacer...
            if len(self.history) > 1:
                # Movemos el estado actual al futuro.
                self.future.append(self.history.pop())
                # Devolvemos el estado anterior.
                return self.history[-1]
            return None

        def redo(self):
            # Si hay algo en el futuro para rehacer...
            if self.future:
                # Movemos el estado rehecho de vuelta al historial.
                state = self.future.pop()
                self.history.append(state)
                # Devolvemos el estado rehecho.
                return state
            return None

    # Creamos una instancia del gestor de estado para la selección de voz.
    voice_state_manager = StateManager()

    # --- 3. CONTROLES DE LA INTERFAZ DE USUARIO (UI) ---

    # Indicador de progreso circular para la carga inicial del modelo.
    loading_indicator = ft.ProgressRing(width=32, height=32, stroke_width=4)
    # Etiqueta de texto que acompaña al indicador de carga.
    loading_label = ft.Text("Cargando modelo de IA, por favor espera...", size=16, weight="bold")

    # Campo de texto principal donde el usuario escribirá.
    text_input = ft.TextField(
        label="Escribe tu texto aquí",
        multiline=True,
        min_lines=4,
        max_lines=8,
        expand=True, # Para que ocupe el espacio disponible.
        disabled=True # Deshabilitado hasta que el modelo cargue.
    )

    # Menú desplegable para seleccionar la voz del locutor.
    speaker_dropdown = ft.Dropdown(
        label="Selecciona una Voz",
        options=[], # Se llenará dinámicamente.
        on_change=None, # Se asignará después.
        disabled=True
    )

    # Botones de Deshacer y Rehacer para los ajustes de voz.
    undo_button = ft.IconButton(icon=ft.icons.UNDO, tooltip="Deshacer cambio de voz", on_click=None, disabled=True)
    redo_button = ft.IconButton(icon=ft.icons.REDO, tooltip="Rehacer cambio de voz", on_click=None, disabled=True)

    # Botón para iniciar la síntesis de texto a voz.
    synthesize_button = ft.ElevatedButton(
        text="Generar Audio",
        icon=ft.icons.SPEAKER_PHONE,
        on_click=None, # La función se asignará después.
        disabled=True
    )

    # Barra de progreso para el proceso de síntesis.
    synthesis_progress = ft.ProgressBar(value=0, width=page.width, visible=False)

    # Creamos un control de imagen que mostrará el gráfico de la onda.
    waveform_plot = ft.Image(visible=False, width=600, height=300, fit=ft.ImageFit.CONTAIN)

    # Botones para la reproducción de audio.
    play_button = ft.IconButton(icon=ft.icons.PLAY_ARROW, on_click=None, tooltip="Reproducir", disabled=True)
    stop_button = ft.IconButton(icon=ft.icons.STOP, on_click=None, tooltip="Detener", disabled=True)

    # Botón para guardar el archivo de audio.
    save_button = ft.ElevatedButton(text="Guardar como .wav", icon=ft.icons.SAVE, on_click=None, disabled=True)

    # Controles para la gestión de configuraciones de voz.
    saved_voices_dropdown = ft.Dropdown(label="Configuraciones Guardadas", on_change=None, options=[])
    save_config_button = ft.IconButton(icon=ft.icons.BOOKMARK_ADD, tooltip="Guardar configuración de voz actual", on_click=None)
    import_configs_button = ft.IconButton(icon=ft.icons.FILE_UPLOAD, tooltip="Importar configuraciones", on_click=None)
    export_configs_button = ft.IconButton(icon=ft.icons.FILE_DOWNLOAD, tooltip="Exportar configuraciones", on_click=None)

    # --- 4. LÓGICA DE LA APLICACIÓN Y MANEJADORES DE EVENTOS ---

    # Función para actualizar el gráfico de la onda.
    def update_waveform_plot(audio_data, sample_rate):
        # Creamos una figura de Matplotlib para el gráfico.
        fig, ax = plt.subplots(figsize=(8, 4))
        # Creamos un eje de tiempo en segundos.
        time_axis = np.linspace(0., len(audio_data) / sample_rate, num=len(audio_data))
        # Dibujamos la onda.
        ax.plot(time_axis, audio_data, color='deepskyblue')
        # Configuramos las etiquetas y el título.
        ax.set_title("Forma de Onda del Audio", color='gray')
        ax.set_xlabel("Tiempo (s)", color='gray')
        ax.set_ylabel("Amplitud", color='gray')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(colors='gray')
        fig.tight_layout()

        # Nos aseguramos de que el directorio 'assets' exista.
        if not os.path.exists("assets"):
            os.makedirs("assets")

        # Guardamos la figura como una imagen PNG.
        img_path = "assets/waveform.png"
        fig.savefig(img_path)
        # Cerramos la figura para liberar memoria.
        plt.close(fig)

        # Actualizamos el control de imagen para que muestre el nuevo gráfico.
        # Añadimos un timestamp para evitar que Flet use una versión en caché de la imagen.
        waveform_plot.src = f"{img_path}?{time.time()}"
        waveform_plot.visible = True
        page.update()

    # --- Funciones para la gestión de configuraciones ---

    # Actualiza el menú desplegable de configuraciones guardadas.
    def update_saved_voices_dropdown():
        options = [ft.dropdown.Option(name) for name in saved_voice_configs.keys()]
        saved_voices_dropdown.options = options
        page.update()

    # Guarda una nueva configuración de voz.
    def save_voice_config(e):
        # Diálogo para pedir el nombre de la configuración.
        config_name_field = ft.TextField(label="Nombre para esta configuración de voz", value=f"Voz {len(saved_voice_configs) + 1}")

        def on_dialog_confirm(e):
            name = config_name_field.value
            if name and name not in saved_voice_configs:
                # Guardamos el locutor actual con el nombre dado.
                saved_voice_configs[name] = {"speaker": speaker_dropdown.value}
                update_saved_voices_dropdown()
                dialog.open = False
                page.update()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Guardar Configuración de Voz"),
            content=config_name_field,
            actions=[
                ft.TextButton("Guardar", on_click=on_dialog_confirm),
                ft.TextButton("Cancelar", on_click=lambda e: setattr(dialog, 'open', False) or page.update()),
            ],
        )
        page.dialog = dialog
        dialog.open = True
        page.update()

    # Carga una configuración de voz desde el menú desplegable.
    def load_voice_config(e):
        config_name = saved_voices_dropdown.value
        if config_name in saved_voice_configs:
            speaker_name = saved_voice_configs[config_name]["speaker"]
            # Si el locutor de la configuración existe, lo seleccionamos.
            if speaker_name in [opt.key for opt in speaker_dropdown.options]:
                speaker_dropdown.value = speaker_name
                # Añadimos este cambio al historial.
                voice_state_manager.add_state(speaker_name)
                page.update()

    # --- Funciones para Deshacer/Rehacer ---

    # Se llama cuando el usuario cambia la voz manualmente.
    def on_speaker_change(e):
        # Añade el nuevo estado al gestor.
        voice_state_manager.add_state(speaker_dropdown.value)
        # Habilita/deshabilita los botones.
        undo_button.disabled = len(voice_state_manager.history) <= 1
        redo_button.disabled = len(voice_state_manager.future) == 0
        page.update()

    # Manejador del botón de deshacer.
    def undo_voice_change(e):
        # Obtiene el estado anterior.
        last_state = voice_state_manager.undo()
        if last_state:
            # Actualiza el desplegable sin disparar el evento 'on_change'.
            speaker_dropdown.value = last_state
        # Actualiza el estado de los botones.
        undo_button.disabled = len(voice_state_manager.history) <= 1
        redo_button.disabled = len(voice_state_manager.future) == 0
        page.update()

    # Manejador del botón de rehacer.
    def redo_voice_change(e):
        # Obtiene el estado futuro.
        next_state = voice_state_manager.redo()
        if next_state:
            # Actualiza el desplegable.
            speaker_dropdown.value = next_state
        # Actualiza el estado de los botones.
        undo_button.disabled = len(voice_state_manager.history) <= 1
        redo_button.disabled = len(voice_state_manager.future) == 0
        page.update()

    # Función que se ejecuta en un hilo para reproducir audio.
    def play_audio_thread():
        # Indicamos que la reproducción está activa.
        is_playing.set()
        # Actualizamos el estado de los botones.
        play_button.disabled = True
        stop_button.disabled = False
        page.update()

        # Usamos sounddevice para reproducir el audio. 'wait' bloquea este hilo, pero no la UI.
        sd.play(audio_cache["data"], audio_cache["rate"])
        sd.wait()

        # Cuando termina, reseteamos el estado.
        is_playing.clear()
        play_button.disabled = False
        stop_button.disabled = True
        page.update()

    # Manejador del botón de reproducir.
    def start_playback(e):
        # Si hay audio y no se está reproduciendo, iniciamos un nuevo hilo de reproducción.
        if audio_cache["data"] is not None and not is_playing.is_set():
            threading.Thread(target=play_audio_thread).start()

    # Manejador del botón de detener.
    def stop_playback(e):
        # Si se está reproduciendo, lo detenemos.
        if is_playing.is_set():
            sd.stop()

    # Función que se ejecuta en un hilo para la síntesis de voz.
    def synthesize_thread(e):
        # Deshabilitamos los controles para evitar acciones conflictivas.
        text_input.disabled = True
        speaker_dropdown.disabled = True
        synthesize_button.disabled = True
        synthesis_progress.visible = True
        play_button.disabled = True
        stop_button.disabled = True
        save_button.disabled = True
        page.update()

        # Obtenemos el texto y el locutor seleccionado.
        text = text_input.value
        speaker = speaker_dropdown.value

        # Si no hay texto, mostramos un error y volvemos al estado normal.
        if not text.strip():
            page.snack_bar = ft.SnackBar(ft.Text("El campo de texto no puede estar vacío."), bgcolor=ft.colors.ERROR)
            page.snack_bar.open = True
            text_input.disabled = False
            speaker_dropdown.disabled = False
            synthesize_button.disabled = False
            synthesis_progress.visible = False
            page.update()
            return

        # Realizamos la síntesis.
        audio_data, sample_rate = synthesizer.synthesize(text, speaker)

        # Guardamos el resultado en nuestra caché.
        audio_cache["data"] = audio_data
        audio_cache["rate"] = sample_rate

        # Actualizamos la visualización de la onda.
        update_waveform_plot(audio_data, sample_rate)

        # Habilitamos los controles correspondientes.
        text_input.disabled = False
        speaker_dropdown.disabled = False
        synthesize_button.disabled = False
        synthesis_progress.visible = False
        play_button.disabled = False
        save_button.disabled = False
        page.update()

    # Manejador del botón de generar.
    def start_synthesis(e):
        # Inicia el proceso de síntesis en un nuevo hilo.
        threading.Thread(target=synthesize_thread, args=(e,)).start()

    # Función que se ejecuta cuando el diálogo de guardado se cierra.
    def save_file_result(e: ft.FilePickerResultEvent):
        # Obtenemos la ruta del archivo seleccionada por el usuario.
        target_path = e.path
        # Si el usuario no canceló, procedemos a guardar.
        if target_path:
            # Aseguramos que la extensión sea .wav.
            if not target_path.lower().endswith(".wav"):
                target_path += ".wav"
            
            # Guardamos el archivo usando el método de nuestro sintetizador.
            success = synthesizer.save_to_wav(audio_cache["data"], audio_cache["rate"], target_path)
            
            # Mostramos un mensaje de confirmación o de error.
            if success:
                page.snack_bar = ft.SnackBar(ft.Text(f"Archivo guardado en: {target_path}"), bgcolor=ft.colors.GREEN_700)
            else:
                page.snack_bar = ft.SnackBar(ft.Text("Hubo un error al guardar el archivo."), bgcolor=ft.colors.ERROR)
            page.snack_bar.open = True
            page.update()

    # Creamos el diálogo de guardado de archivos.
    file_picker = ft.FilePicker(on_result=save_file_result)
    # Lo añadimos a la superposición de la página para que sea visible.
    page.overlay.append(file_picker)

    # --- Funciones para Importar/Exportar ---
    def on_import_result(e: ft.FilePickerResultEvent):
        if e.files:
            import_path = e.files[0].path
            try:
                with open(import_path, "r", encoding="utf-8") as f:
                    # Cargamos las configuraciones del archivo JSON.
                    imported_configs = json.load(f)
                    # Las fusionamos con las existentes.
                    saved_voice_configs.update(imported_configs)
                    update_saved_voices_dropdown()
                    page.snack_bar = ft.SnackBar(ft.Text("Configuraciones importadas con éxito."), bgcolor=ft.colors.GREEN_700)
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"Error al importar: {ex}"), bgcolor=ft.colors.ERROR)
            page.snack_bar.open = True
            page.update()

    def on_export_result(e: ft.FilePickerResultEvent):
        if e.path:
            export_path = e.path
            if not export_path.lower().endswith(".json"):
                export_path += ".json"
            try:
                with open(export_path, "w", encoding="utf-8") as f:
                    # Guardamos el diccionario de configuraciones en un archivo JSON.
                    json.dump(saved_voice_configs, f, indent=4)
                page.snack_bar = ft.SnackBar(ft.Text(f"Configuraciones exportadas a: {export_path}"), bgcolor=ft.colors.GREEN_700)
            except Exception as ex:
                page.snack_bar = ft.SnackBar(ft.Text(f"Error al exportar: {ex}"), bgcolor=ft.colors.ERROR)
            page.snack_bar.open = True
            page.update()

    import_picker = ft.FilePicker(on_result=on_import_result)
    export_picker = ft.FilePicker(on_result=on_export_result)
    page.overlay.extend([import_picker, export_picker])


    # --- 5. INICIALIZACIÓN DE LA APLICACIÓN ---

    # Función que se ejecuta en un hilo para inicializar el sintetizador.
    def initialize_synthesizer():
        nonlocal synthesizer
        # Creamos la instancia de la clase Synthesizer.
        synthesizer = Synthesizer()

        # Una vez cargado, obtenemos los locutores y los añadimos al menú desplegable.
        speakers = synthesizer.get_speakers()
        speaker_options = [ft.dropdown.Option(s) for s in speakers]
        speaker_dropdown.options = speaker_options
        # Seleccionamos el primer locutor por defecto.
        if speakers:
            initial_speaker = speakers[0]
            speaker_dropdown.value = initial_speaker
            # Guardamos el estado inicial para el sistema de Deshacer.
            voice_state_manager.add_state(initial_speaker)
            undo_button.disabled = False # Se puede deshacer al menos una vez al estado inicial.

        # Ocultamos el indicador de carga.
        loading_indicator.visible = False
        loading_label.visible = False

        # Habilitamos los controles principales de la UI.
        text_input.disabled = False
        speaker_dropdown.disabled = False
        synthesize_button.disabled = False
        page.update()

    # Asignamos los manejadores de eventos a los botones.
    speaker_dropdown.on_change = on_speaker_change
    undo_button.on_click = undo_voice_change
    redo_button.on_click = redo_voice_change
    synthesize_button.on_click = start_synthesis
    play_button.on_click = start_playback
    stop_button.on_click = stop_playback
    save_button.on_click = lambda _: file_picker.save_file(
        dialog_title="Guardar archivo de audio",
        file_name="audio.wav",
        allowed_extensions=["wav"]
    )
    save_config_button.on_click = save_voice_config
    saved_voices_dropdown.on_change = load_voice_config
    import_configs_button.on_click = lambda _: import_picker.pick_files(allow_multiple=False, allowed_extensions=["json"])
    export_configs_button.on_click = lambda _: export_picker.save_file(file_name="voxflow_configs.json", allowed_extensions=["json"])

    # --- 6. DISEÑO Y ESTRUCTURA DE LA PÁGINA ---

    # Añadimos los controles iniciales de carga a la página.
    page.add(
        ft.Column(
            [loading_indicator, loading_label],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            height=page.height,
            width=page.width
        )
    )

    # Añadimos el resto de la interfaz (inicialmente vacía hasta que cargue el modelo).
    page.add(
        ft.Column([
            # Fila superior con el campo de texto y los controles de voz.
            ft.Row([
                text_input,
                ft.Column([
                    ft.Row([speaker_dropdown, undo_button, redo_button]),
                    ft.Text("Gestión de Configuraciones", weight="bold"),
                    ft.Row([saved_voices_dropdown, save_config_button]),
                    ft.Row([import_configs_button, export_configs_button])
                ], spacing=5)
            ], alignment=ft.MainAxisAlignment.START),

            # Controles principales y visualización.
            synthesize_button,
            synthesis_progress,
            waveform_plot,
            ft.Row([play_button, stop_button], alignment=ft.MainAxisAlignment.CENTER),
            save_button
        ], alignment=ft.MainAxisAlignment.START, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=15)
    )

    # Iniciamos la carga del modelo en un hilo separado para no bloquear la UI.
    threading.Thread(target=initialize_synthesizer).start()

# Punto de entrada para ejecutar la aplicación.
if __name__ == "__main__":
    ft.app(target=main)
