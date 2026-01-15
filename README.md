# 🎙️ VoxFlow v2.1: Natural TTS con IA Integrada y CustomTkinter

**VoxFlow** es una aplicación de escritorio para la síntesis de voz (Text-to-Speech) de alta calidad, impulsada por el motor de inteligencia artificial de **Coqui TTS** y construida con la moderna librería de interfaz gráfica **CustomTkinter**.

---

## ⚠️ Requisitos Obligatorios

-   **Python 3.10 (requerido)**. Se recomienda el uso de un entorno virtual (`venv`).
-   Este proyecto utiliza Coqui TTS (`TTS`), que **no es compatible con versiones de Python 3.12 o superiores**. Asegúrate de usar una versión de Python 3.10 para evitar problemas de instalación.

---

## ✨ Características Principales

- **Motor de IA Avanzado:** Utiliza `xtts_v2`, un modelo de voz de última generación de Coqui TTS para una claridad y naturalidad excepcionales, con capacidad de clonación de voz.
- **Selección de Voces por Referencia:** Elige entre voces predefinidas (masculina, femenina, infantil) que sirven como referencia para la clonación de voz. ¡Puedes añadir tus propios archivos `.wav` en la carpeta `assets/voices` para crear nuevas voces!
- **Multiplataforma:** Diseñada para funcionar en **Windows, macOS y Linux**.
- **Interfaz Gráfica Moderna:** Una UI intuitiva y estéticamente agradable construida con **CustomTkinter**.
- **Funcionalidades Avanzadas:**
  - **Visualizador de Onda:** Analiza la forma de onda del audio generado en tiempo real.
  - **Reproducción Instantánea:** Escucha el audio con controles de **Play/Stop** sin necesidad de guardarlo primero.
  - **Guardado Personalizado:** Guarda el resultado como un archivo **`.wav`** donde prefieras.
  - **Gestión de Sesión:** **Guarda y Carga** tu texto y voz seleccionada para retomar tu trabajo fácilmente.
- **Privacidad Total:** El procesamiento se realiza 100% en tu dispositivo.

---

## 🛠️ Instalación y Configuración

### 1. Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/VoxFlow.git
cd VoxFlow
```

### 2. Configurar un Entorno Virtual con Python 3.10

Este paso es crucial. Para evitar conflictos con otras versiones de Python que puedas tener en tu sistema, te recomendamos encarecidamente crear un entorno virtual que utilice específicamente Python 3.10.

**Si ya tienes Python 3.10 instalado:**

*   **En macOS / Linux:**
    ```bash
    # Crea el entorno virtual llamado '.venv' usando el ejecutable de python3.10
    python3.10 -m venv .venv

    # Activa el entorno
    source .venv/bin/activate
    ```

*   **En Windows (PowerShell):**
    ```powershell
    # Asumiendo que 'py -3.10' apunta a tu instalación de Python 3.10
    py -3.10 -m venv .venv

    # Activa el entorno
    .venv\Scripts\Activate.ps1
    ```
    *Si usas el Símbolo del sistema (CMD), el comando de activación es `.venv\Scripts\activate.bat`.*

Una vez activado, verás `(.venv)` al principio de la línea de tu terminal. Esto confirma que cualquier paquete de Python que instales quedará aislado en este proyecto.

### 3. Instalar las Dependencias
Con el entorno virtual activado, instala todos los paquetes necesarios con un solo comando:
```bash
pip install -r requirements.txt
```
*Nota: La primera vez que instales `TTS`, se descargarán varias dependencias, incluyendo PyTorch, lo cual puede tardar unos minutos.*

---

## 💻 ¿Cómo Usar la Aplicación?

Una vez completada la instalación, ejecuta la aplicación con:
```bash
python main.py
```
La primera vez que ejecutes la aplicación, Coqui TTS descargará automáticamente el modelo de voz `xtts_v2`. Este proceso puede tardar un poco y requiere conexión a internet. Las siguientes veces, la aplicación se iniciará mucho más rápido.

En la ventana principal podrás:
1.  **Esperar a que cargue el modelo** (verás el mensaje "Motor IA listo para usar").
2.  **Escribir o pegar texto** en el área designada.
3.  **Seleccionar una voz** ("Voz Masculina", "Voz Femenina", etc.) en el menú desplegable.
4.  Pulsar **"Generar Audio"**. La forma de onda aparecerá en el visor.
5.  Usar los botones **Play/Stop** para escuchar el resultado.
6.  Pulsar **"Guardar Audio"** para guardar el archivo `.wav`.
7.  Utilizar **"Cargar/Guardar Config"** para gestionar tu sesión.

---

## 📂 Estructura del Proyecto

-   `main.py`: Contiene todo el código de la interfaz de usuario con **CustomTkinter**. Gestiona los eventos, la disposición de los widgets y la interacción con el usuario de forma segura entre hilos.
-   `voxflow_core.py`: El cerebro del proyecto. La clase `Synthesizer` inicializa Coqui TTS, gestiona la carga de modelos, la clonación de voz a partir de archivos de referencia y el guardado de audio.
-   `requirements.txt`: Lista todas las dependencias de Python compatibles con Python 3.10.
-   `assets/voices/`: Carpeta que contiene los archivos `.wav` de referencia para la clonación de voz.
-   `.github/workflows/tests.yml`: Define el flujo de trabajo de Integración Continua (CI) en GitHub Actions, configurado para usar Python 3.10.

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Si tienes ideas para mejorar la aplicación, no dudes en abrir un Pull Request o una Issue.
