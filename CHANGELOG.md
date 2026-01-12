# Historial de Cambios (Changelog)

Todas las modificaciones notables de este proyecto serán documentadas en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
y este proyecto se adhiere al [Versionamiento Semántico](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-12

### Añadido (`Added`)

-   **Lanzamiento inicial de VoxFlow.**
-   Funcionalidad principal de Texto-a-Voz para Windows, macOS y Linux usando el motor Piper.
-   Interfaz gráfica de usuario (GUI) de escritorio construida con el framework Flet.
-   Capacidad para reproducir audio directamente desde un texto introducido.
-   Función para cargar texto desde un archivo `.txt` local.
-   Función para guardar el audio generado como un archivo `.mp3`.
-   Indicador de progreso (`ProgressRing`) que ofrece feedback visual durante la síntesis de audio.
-   Código fuente (`main.py` y `voxflow_core.py`) completamente comentado línea por línea en español.
-   Documentación exhaustiva en `README.md` con instrucciones de instalación y uso.
-   Archivo `requirements.txt` para gestionar las dependencias de Python.
-   Archivo `.gitignore` para mantener el repositorio limpio de archivos innecesarios.

### Cambiado (`Changed`)

-   **Refactorización de la Arquitectura:** El proyecto fue reestructurado para separar la lógica de la interfaz de usuario (`main.py`) de la lógica de negocio principal (`voxflow_core.py`), mejorando la mantenibilidad.
-   **Mejora de Seguridad (Crítica):** Se eliminó una vulnerabilidad de inyección de comandos. La entrada de texto del usuario ahora se pasa de forma segura a los procesos externos (`piper`, `ffmpeg`) a través de `stdin`, en lugar de formatear cadenas de texto en un comando `shell`.
-   **Soporte para macOS Mejorado:** La reproducción de audio en macOS ahora utiliza el reproductor nativo `afplay` en lugar de `aplay` para una mejor compatibilidad y menos dependencias.

### Dependencias

-   El proyecto depende de `flet` para la interfaz gráfica.
-   Requiere que el usuario tenga las dependencias externas `piper` y `ffmpeg` instaladas y disponibles en el PATH del sistema.
