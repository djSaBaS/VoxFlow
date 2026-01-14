# Historial de Cambios (Changelog)

Todas las modificaciones notables de este proyecto serán documentadas en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
y este proyecto se adhiere al [Versionamiento Semántico](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2024-07-29

### Añadido (`Added`)

-   **Cambio de Motor a Coqui TTS:** El motor de síntesis de voz ha sido migrado de `Piper` a `Coqui TTS`, eliminando la necesidad de dependencias externas como `piper` y `ffmpeg`.
-   **Visualizador de Ondas:** Se ha añadido un gráfico que muestra la forma de onda del audio generado.
-   **Reproducción en la Aplicación:** Se han añadido botones de "Reproducir" y "Detener" para escuchar el audio directamente en la interfaz sin necesidad de guardarlo.
-   **Gestión Avanzada de Configuraciones de Voz:**
    -   Funcionalidad para **Guardar** y **Cargar** configuraciones de voz personalizadas.
    -   Capacidad para **Importar** y **Exportar** configuraciones en formato `.json`.
    -   Botones de **Deshacer** y **Rehacer** para gestionar los cambios en la configuración.
-   **Selección de Locutor:** Un menú desplegable permite al usuario elegir entre las diferentes voces disponibles en el modelo TTS.
-   **Barras de Progreso Detalladas:** Se muestran indicadores de progreso tanto para la carga del modelo inicial como para la síntesis de voz.
-   **Guardado de Archivo Personalizado:** El usuario puede ahora especificar el nombre y la ubicación del archivo `.wav` a guardar mediante un diálogo nativo del sistema.
-   **Tests Unitarios y de Integración:** Se ha añadido un conjunto de pruebas utilizando `pytest` y `unittest.mock` para validar la lógica del núcleo de la aplicación.
-   **Integración Continua (CI):** Se ha configurado un flujo de trabajo de GitHub Actions (`.github/workflows/ci.yml`) para ejecutar las pruebas automáticamente.
-   **Scripts de Auditoría de API:** Se han añadido scripts para listar iconos y auditar la compatibilidad de la API de Flet, ayudando al mantenimiento futuro.

### Cambiado (`Changed`)

-   **Refactorización Completa del Código:** `main.py` y `voxflow_core.py` han sido reescritos para adaptarse a la nueva arquitectura basada en `Coqui TTS` y mejorar la estabilidad.
-   **Formato de Salida:** El formato de audio por defecto ha cambiado de `.mp3` a `.wav` para eliminar la dependencia de `ffmpeg`.
-   **Interfaz de Usuario Renovada:** La interfaz ha sido rediseñada para incorporar las nuevas funcionalidades de manera más limpia e intuitiva.

### Corregido (`Fixed`)

-   **Estabilidad Crítica de la Aplicación:** Se han solucionado numerosos errores de `TypeError` y `AttributeError` relacionados con incompatibilidades de la API de Flet. La aplicación ahora se inicia y funciona de manera estable.
-   **Compatibilidad de Controles Flet:** Se han actualizado todos los controles de la interfaz de usuario (botones, desplegables, iconos, imágenes) para usar la sintaxis moderna y compatible, resolviendo los problemas que impedían el arranque de la aplicación.
-   **Llamada de Inicio de Flet:** Se ha actualizado el inicio de la aplicación de `ft.app()` a `ft.run()`, siguiendo las recomendaciones de la biblioteca.

### Eliminado (`Removed`)

-   **Dependencias Externas:** Se ha eliminado por completo la dependencia de los ejecutables `piper` y `ffmpeg`, haciendo la aplicación más autocontenida y fácil de distribuir.
-   **Sistema de Logging:** Se eliminó temporalmente el sistema de `logging` debido a conflictos complejos con el subproceso de Flet, priorizando la estabilidad de la aplicación.

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
