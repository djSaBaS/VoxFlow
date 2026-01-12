import flet as ft # Importamos el framework para la interfaz de usuario
import subprocess # Para ejecutar el proceso de Piper en el sistema
import os          # Para manejar rutas de archivos de forma segura

def main(page: ft.Page):
    # Configuramos el título y el tema de la ventana de la aplicación
    page.title = "Natural TTS Multiplataforma"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20

    # Variable para almacenar la ruta del modelo de voz (archivo .onnx)
    # Debes descargar un modelo de https://github.com/rhasspy/piper/releases
    MODEL_PATH = "es_ES-sharvard-medium.onnx"

    # Definimos el campo de entrada de texto donde el usuario escribirá
    text_input = ft.TextField(
        label="Escribe el texto aquí",
        multiline=True,
        min_lines=3,
        placeholder="Hola, ¿cómo estás hoy?"
    )

    # Función que se ejecuta al presionar el botón de "Convertir"
    def speak_text(e):
        # Verificamos si hay texto en el campo de entrada
        if not text_input.value:
            page.snack_bar = ft.SnackBar(ft.Text("Por favor, ingresa un texto"))
            page.snack_bar.open = True
            page.update()
            return

        try:
            # Comando para ejecutar Piper:
            # 1. 'echo' envía el texto al proceso
            # 2. 'piper' procesa el texto y lo reproduce por los altavoces
            # Nota: En Windows, Linux y Mac, Piper puede enviar el audio directamente a 'aplay' o 'ffplay'
            command = f'echo "{text_input.value}" | piper --model {MODEL_PATH} --output_raw | aplay -r 22050 -f S16_LE -t raw'
            
            # Ejecutamos el comando en el sistema operativo de forma asíncrona
            subprocess.Popen(command, shell=True)
            
        except Exception as ex:
            # Si ocurre un error, lo mostramos en una notificación
            page.snack_bar = ft.SnackBar(ft.Text(f"Error: {str(ex)}"))
            page.snack_bar.open = True
            page.update()

    # Creamos el botón con un icono de reproducción
    play_button = ft.ElevatedButton(
        "Convertir a Voz Natural",
        icon=ft.icons.PLAY_ARROW,
        on_click=speak_text
    )

    # Añadimos los elementos visuales a la página de la aplicación
    page.add(
        ft.Text("Convertidor de Texto a Voz (Multiplataforma)", size=24, weight="bold"),
        text_input,
        play_button
    )

# Punto de entrada para ejecutar la aplicación
if __name__ == "__main__":
    # ft.app inicia la aplicación; target=main define la función principal
    # Para móviles se puede usar view=ft.AppView.WEB_BROWSER durante pruebas
    ft.app(target=main)
