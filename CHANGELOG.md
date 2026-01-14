# Historial de Cambios (Changelog)

Todas las modificaciones notables de este proyecto serán documentadas en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
y este proyecto se adhiere al [Versionamiento Semántico](https://semver.org/spec/v2.0.0.html).

## [2.0.6] - 2024-07-29

### Corregido (`Fixed`)

-   **Auditoría Final y Definitiva de Compatibilidad de Flet:** Se ha realizado una auditoría proactiva y exhaustiva de todos los controles de Flet para garantizar la compatibilidad total con la versión de la API utilizada. Se han corregido todos los `TypeError` y `AttributeError` restantes, incluyendo los de `FilePicker` y las propiedades de alineación. La aplicación es ahora completamente estable y se inicia sin errores.

## [2.0.5] - 2024-07-29

### Corregido (`Fixed`)

-   **Auditoría Final de Compatibilidad de Flet:** Se ha realizado una revisión exhaustiva final de todos los controles de la interfaz de usuario para garantizar la compatibilidad con la versión de Flet utilizada. Se ha corregido un `TypeError` en `ft.Image` al añadir el argumento `src` requerido en su inicialización. Con esta corrección, la aplicación es ahora completamente estable y se inicia correctamente.

## [2.0.4] - 2024-07-29

### Corregido (`Fixed`)

-   **Auditoría de Compatibilidad de Flet:** Se ha realizado una revisión exhaustiva de todos los controles de la interfaz de usuario para garantizar la compatibilidad con la versión de Flet utilizada. Se han corregido múltiples `AttributeError` y `TypeError` (en `ft.ImageFit`, `ft.icons`, `ft.ElevatedButton`, etc.) utilizando sus equivalentes en formato de cadena de texto o la sintaxis de argumentos moderna. La aplicación es ahora completamente estable.

## [2.0.3] - 2024-07-29

### Corregido (`Fixed`)

-   **Error de `TypeError` en `ElevatedButton`:** Se ha corregido un error de compatibilidad con la API de Flet, reemplazando el argumento obsoleto `text` por el argumento `content` con un control `ft.Text` en todos los botones de la aplicación.

## [2.0.2] - 2024-07-29

### Corregido (`Fixed`)

-   **Error de `AttributeError` en Iconos:** Se ha corregido un error que impedía iniciar la aplicación debido a la forma en que se referenciaban los iconos. Todas las llamadas a iconos (`ft.icons.NOMBRE`) han sido reemplazadas por su equivalente en formato de cadena de texto (`"NOMBRE"`) para garantizar la máxima compatibilidad entre diferentes versiones de Flet.

## [2.0.1] - 2024-07-29

### Corregido (`Fixed`)

-   **Error de `TypeError` en `Dropdown`:** Se ha corregido un error de compatibilidad con la versión de Flet que impedía iniciar la aplicación, moviendo la asignación del evento `on_change` para que ocurra después de la inicialización del control.
-   **Advertencia `DeprecationWarning`:** Se ha actualizado la llamada de inicio de la aplicación de `ft.app()` a `ft.run()`, siguiendo las recomendaciones de las versiones más recientes de Flet.

### Añadido (`Added`)

-   **Sistema de Registro de Logs:** Se ha implementado un sistema de registro (`logging`) que guarda los eventos importantes y los errores en un archivo `voxflow.log`. Esto mejora enormemente la capacidad de diagnosticar problemas futuros.

## [2.0.0] - 2024-07-29

### Añadido (`Added`)

-   **Visualizador de Ondas:** Se ha añadido un gráfico que muestra la forma de onda del audio generado en tiempo real.
-   **Reproducción Directa:** Ahora se puede reproducir y detener el audio directamente en la aplicación sin necesidad de guardarlo.
-   **Selección de Voz:** Se ha implementado un menú desplegable para elegir entre múltiples voces (locutores) disponibles en el modelo de IA.
-   **Gestión de Configuraciones de Voz:**
    -   Funcionalidad para **Guardar** y **Cargar** configuraciones de voz personalizadas.
    -   Capacidad para **Importar** y **Exportar** configuraciones en formato `.json`.
    -   Botones de **Deshacer** y **Rehacer** para la selección de voz.
-   **Barras de Progreso:** Indicadores visuales para la carga del modelo y la síntesis de voz, mejorando la experiencia de usuario.
-   **Diálogo de Guardado Personalizado:** El usuario ahora puede elegir el nombre y la ubicación del archivo `.wav` de salida.

### Cambiado (`Changed`)

-   **Cambio de Motor de TTS (Crítico):** Se ha reemplazado el motor `Piper` por **`Coqui TTS`**. La aplicación ahora utiliza una biblioteca de Python pura, eliminando la necesidad de que los usuarios instalen dependencias externas.
-   **Refactorización Completa:** El código de `main.py` y `voxflow_core.py` ha sido reescrito para soportar la nueva arquitectura basada en Coqui TTS, mejorando la modularidad y la mantenibilidad.
-   **Formato de Salida:** La aplicación ahora genera archivos en formato `.wav` en lugar de `.mp3` para simplificar el proceso y eliminar la dependencia de FFmpeg.
-   **Interfaz de Usuario Renovada:** La interfaz ha sido rediseñada para acomodar todas las nuevas funcionalidades de forma intuitiva.

### Eliminado (`Removed`)

-   **Dependencias Externas:** Se ha eliminado por completo la dependencia de los ejecutables externos `piper` y `ffmpeg`. La aplicación ahora es mucho más fácil de instalar y distribuir.

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
