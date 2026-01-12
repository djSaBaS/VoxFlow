# 🎙️ VoxFlow v1.0: Natural Multiplatform TTS

**VoxFlow** es una solución de síntesis de voz (Text-to-Speech) que proporciona voces naturales y fluidas en múltiples plataformas. Basada en el motor de inferencia de Piper, esta aplicación te permite convertir texto a voz de alta calidad de forma local y privada.

---

## ✨ Características Principales

- **Naturalidad Superior:** Utiliza los modelos de voz de última generación de Piper para una entonación y claridad casi humanas.
- **Multiplataforma (Escritorio):** Diseñado para funcionar en **Windows, macOS y Linux**. El soporte para **Android** es un objetivo a futuro.
- **Interfaz Gráfica Sencilla:** Una aplicación de escritorio intuitiva construida con Flet.
- **Funcionalidades Clave:**
  - Pega texto directamente o **carga archivos `.txt`**.
  - **Reproduce el audio** al instante.
  - **Guarda la salida como un archivo `.mp3`** para usarla donde quieras.
- **Privacidad Total:** Todo el procesamiento se realiza en tu dispositivo. No se envían datos a la nube.
- **Alto Rendimiento:** Optimizado para funcionar de manera eficiente incluso en hardware modesto.

---

## 🛠️ Instalación y Configuración

Para poner en marcha la aplicación, sigue estos pasos:

### 1. Requisitos Previos (¡Importante!)

Antes de ejecutar la aplicación, necesitas instalar dos herramientas externas:

- **Piper:** Es el motor de síntesis de voz. Descárgalo desde su [página oficial de GitHub](https://github.com/rhasspy/piper/releases). Debes descargar el binario correspondiente a tu sistema operativo y añadirlo al **PATH** del sistema para que la aplicación pueda encontrarlo.
- **FFmpeg:** Es una herramienta esencial para la manipulación de audio y video. La usamos para convertir el audio generado a formato MP3.
  - **Windows:** Descárgalo desde [su sitio web oficial](https://ffmpeg.org/download.html) y añade la carpeta `bin` a tu PATH.
  - **macOS (con Homebrew):** `brew install ffmpeg`
  - **Linux (Debian/Ubuntu):** `sudo apt-get install ffmpeg`

### 2. Configuración del Proyecto

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/tu-usuario/VoxFlow.git
    cd VoxFlow
    ```

2.  **Instalar las dependencias de Python:**
    Se recomienda crear un entorno virtual para mantener las dependencias aisladas.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # En Windows: .venv\Scripts\activate
    ```
    Luego, instala los paquetes necesarios:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Descargar un Modelo de Voz:**
    La aplicación necesita un modelo de voz en formato `.onnx`. Descarga el modelo en español `es_ES-sharvard-medium.onnx` desde [aquí](https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx) y colócalo en la carpeta raíz del proyecto.

---

## 💻 ¿Cómo Usar la Aplicación?

Una vez que hayas completado la instalación, ejecuta la aplicación con:
```bash
python3 main.py
```
Se abrirá una ventana donde podrás:
1.  **Escribir o pegar texto** en el área designada.
2.  Hacer clic en **"Subir Archivo .txt"** para cargar texto desde un archivo.
3.  Pulsar **"Convertir a Voz"** para escuchar el resultado.
4.  Pulsar **"Guardar como MP3"** para guardar el audio en tu disco.

---

## 📂 Estructura del Proyecto

El código está organizado de forma limpia para separar la lógica de la interfaz:

-   `main.py`: Contiene todo el código de la interfaz de usuario creada con Flet. Gestiona los botones, campos de texto y eventos.
-   `voxflow_core.py`: El cerebro del proyecto. La clase `Synthesizer` se encarga de interactuar con Piper y FFmpeg para generar y guardar el audio, con lógica adaptada para cada sistema operativo.
-   `requirements.txt`: Lista las dependencias de Python.
-   `CHANGELOG.md`: Historial de cambios y versiones del proyecto.

---

## 📦 Creación de Ejecutables

Si deseas distribuir esta aplicación como un ejecutable independiente para que los usuarios no necesiten instalar Python, puedes usar el comando `build` de Flet.

Desde la carpeta raíz del proyecto, ejecuta el comando correspondiente a tu sistema operativo de destino:

-   **Para Windows:**
    ```bash
    flet build windows
    ```
-   **Para macOS:**
    ```bash
    flet build macos
    ```
-   **Para Linux:**
    ```bash
    flet build linux
    ```

El ejecutable resultante se encontrará en la carpeta `build/`.

> **⚠️ Nota Importante sobre las Dependencias Externas:**
> La compilación de la aplicación **NO** incluye las herramientas `piper` y `ffmpeg`. Esto significa que **el usuario final todavía necesita instalar `piper` y `ffmpeg` por separado** en su sistema y asegurarse de que estén accesibles en el PATH para que la aplicación funcione correctamente.

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si tienes ideas para mejorar la aplicación, optimizar el rendimiento o añadir nuevas funcionalidades, no dudes en abrir un Pull Request o una Issue.
