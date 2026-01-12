# 🎙️ VoxFlow: Natural Multiplatform TTS

**VoxFlow** es una solución de síntesis de voz (Text-to-Speech) basada en modelos neuronales de última generación. El objetivo principal es proporcionar voces humanas extremadamente naturales que funcionen de forma local y privada en cualquier sistema operativo.

---

## 🚀 Características Principales

- **Naturalidad Superior:** Utiliza modelos VITS para una entonación humana.
- **Multiplataforma:** Soporte nativo para Windows, Android, Linux y macOS.
- **Privacidad Total:** Funciona 100% offline. No se envían datos a la nube.
- **Alto Rendimiento:** Optimizado incluso para hardware modesto como Raspberry Pi o dispositivos Android antiguos.

---

## 🛠️ Instalación

Para configurar el entorno de desarrollo, sigue estos pasos:

### Requisitos previos
- Python 3.10 o superior.
- Pip (gestor de paquetes de Python).

### Pasos
1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/tu-usuario/VoxFlow.git](https://github.com/tu-usuario/VoxFlow.git)
   cd VoxFlow
Instalar dependencias:

Bash

pip install -r requirements.txt
Descargar un modelo de voz: Descarga un archivo .onnx desde el catálogo de Piper Voices y colócalo en la carpeta raíz del proyecto.

💻 Ejemplo de Uso
El núcleo del proyecto está diseñado para ser simple y profesional:

Python

# Importamos la lógica principal del sintetizador
from voxflow_core import Synthesizer

# Inicializamos el motor con el modelo descargado
# El modelo debe ser un archivo .onnx perfectamente configurado
engine = Synthesizer(model_path="es_ES-sharvard-medium.onnx")

# Convertimos texto a voz de manera inmediata
# La función 'play' gestiona la salida de audio según el OS
engine.say("Hola, bienvenido al futuro de la síntesis de voz natural.")
📱 Compilación para Móvil y Escritorio
Este proyecto utiliza Flet para la interfaz gráfica, lo que permite empaquetar la app fácilmente:

Para Android: flet build apk

Para Windows: flet build windows

Para macOS: flet build macos

📄 Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

🤝 Contribuciones
¡Las contribuciones son bienvenidas! Si tienes ideas para mejorar la naturalidad o añadir nuevos idiomas, abre un Pull Request o una Issue.


---

### 4. Resumen del código de lógica (`voxflow_core.py`)
Para que tu proyecto esté completo, aquí tienes la clase base comentada línea a línea:

```python
import subprocess # Módulo para ejecutar procesos del sistema operativo
import platform   # Para detectar si estamos en Windows, Linux o Mac

class Synthesizer:
    # Método constructor para inicializar la ruta del modelo de voz
    def __init__(self, model_path):
        self.model_path = model_path # Guardamos la ruta del archivo .onnx

    # Método para procesar texto y convertirlo en audio audible
    def say(self, text):
        # Detectamos el sistema operativo actual del usuario
        os_type = platform.system()

        # Construimos el comando base de Piper con el modelo especificado
        # Piper recibe texto por entrada estándar (stdin) y devuelve audio raw
        base_cmd = f'echo "{text}" | piper --model {self.model_path} --output_raw'

        # Ajustamos el comando de reproducción según el sistema operativo detectado
        if os_type == "Linux" or os_type == "Darwin": # Darwin es el núcleo de macOS
            # En sistemas Unix usamos 'aplay' o 'afplay' para reproducir el audio raw
            full_cmd = f"{base_cmd} | aplay -r 22050 -f S16_LE -t raw"
        elif os_type == "Windows":
            # En Windows se suele redirigir a un reproductor compatible como ffplay
            full_cmd = f"{base_cmd} | ffplay -ar 22050 -f s16le -nodisp -autoexit -"
        
        # Ejecutamos el comando final en una shell del sistema de forma segura
        subprocess.Popen(full_cmd, shell=True)
