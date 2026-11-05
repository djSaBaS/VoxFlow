# 🎙️ VoxFlow v2.0: Natural TTS con IA Integrada

**VoxFlow** es una aplicación de escritorio para la síntesis de voz (Text-to-Speech) de alta calidad, ahora impulsada por el motor de inteligencia artificial de **Coqui TTS**. Esta nueva versión elimina las dependencias externas, ofreciendo una experiencia de usuario más fluida, potente y totalmente autocontenida.

---

## ✨ Características Principales

- **Motor de IA Avanzado:** Utiliza los modelos de voz de última generación de Coqui TTS para una claridad y naturalidad excepcionales.
- **Totalmente Autocontenida:** Ya **no requiere instalar `piper` ni `ffmpeg`**. Toda la funcionalidad está empaquetada dentro de la aplicación Python.
- **Multiplataforma:** Diseñada para funcionar en **Windows, macOS y Linux**.
- **Interfaz Gráfica Intuitiva:** Una UI moderna y fácil de usar construida con Flet.
- **Funcionalidades Avanzadas:**
  - **Selección de Voz Dinámica:** Elige entre una variedad de voces (locutores) disponibles en el modelo.
  - **Visualizador de Onda:** Analiza la forma de onda del audio generado en tiempo real.
  - **Reproducción Instantánea:** Escucha el audio con controles de **Play/Stop** sin necesidad de guardarlo primero.
  - **Guardado Personalizado:** Guarda el resultado como un archivo **`.wav`** con el nombre que elijas.
  - **Gestión de Configuraciones de Voz:**
    - **Guarda y Carga** tus ajustes de voz preferidos.
    - **Exporta e Importa** configuraciones en formato `.json` para compartirlas.
    - **Deshacer y Rehacer** cambios en la selección de voz.
- **Privacidad Total:** El procesamiento se realiza 100% en tu dispositivo.

---

## 🛠️ Instalación y Configuración

Poner en marcha la aplicación ahora es más fácil que nunca.

### 1. Requisitos Previos

-   **Python 3.9+** instalado en tu sistema.

### 2. Configuración del Proyecto

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/djSaBaS/VoxFlow.git
    cd VoxFlow
    ```

2.  **Instalar las dependencias de Python:**
    Se recomienda encarecidamente crear un entorno virtual.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # En Windows: .venv\Scripts\activate
    ```
    Luego, instala todos los paquetes necesarios con un solo comando:
    ```bash
    pip install -r requirements.txt
    ```
    *Nota: La primera vez que instales `coqui-tts`, se descargarán varias dependencias, incluyendo PyTorch, lo cual puede tardar unos minutos.*

---

## 💻 ¿Cómo Usar la Aplicación?

Una vez completada la instalación, ejecuta la aplicación con:
```bash
python3 main.py
```
La primera vez que ejecutes la aplicación, Coqui TTS descargará automáticamente el modelo de voz necesario. Este proceso puede tardar un poco y requiere conexión a internet. Las siguientes veces, la aplicación se iniciará mucho más rápido.

En la ventana principal podrás:
1.  **Esperar a que cargue el modelo** (verás un indicador).
2.  **Escribir o pegar texto** en el área designada.
3.  **Seleccionar una voz** en el menú desplegable.
4.  Pulsar **"Generar Audio"**. La forma de onda aparecerá en el visor.
5.  Usar los botones **Play/Stop** para escuchar el resultado.
6.  Pulsar **"Guardar como .wav"** para guardar el audio.
7.  Utilizar los botones de **gestión de configuraciones** para guardar, cargar, importar o exportar tus voces favoritas.

---

## 📂 Estructura del Proyecto

-   `main.py`: Contiene todo el código de la interfaz de usuario con Flet. Gestiona los eventos, la disposición de los controles y la interacción con el usuario.
-   `voxflow_core.py`: El cerebro del proyecto. La clase `Synthesizer` inicializa Coqui TTS, gestiona la carga de modelos, la síntesis de voz y el guardado de archivos.
-   `requirements.txt`: Lista todas las dependencias de Python.
-   `CHANGELOG.md`: Historial de cambios y versiones del proyecto.

---

## 📦 Creación de Ejecutables

Puedes distribuir esta aplicación como un ejecutable independiente usando el comando `build` de Flet.

-   **Para Windows:** `flet build windows`
-   **Para macOS:** `flet build macos`
-   **Para Linux:** `flet build linux`

> **⚠️ Nota Importante sobre los Modelos de IA:**
> La compilación con Flet empaquetará todas las dependencias de Python. Sin embargo, **el modelo de Coqui TTS no se incluye en el ejecutable**. La primera vez que un usuario final ejecute la aplicación, esta necesitará una conexión a internet para descargar y cachear el modelo de voz. Después de esa primera ejecución, la aplicación podrá funcionar sin conexión.

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Si tienes ideas para mejorar la aplicación, no dudes en abrir un Pull Request o una Issue.
