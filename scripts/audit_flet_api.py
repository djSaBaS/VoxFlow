# -*- coding: utf-8 -*-
# Script para auditar la compatibilidad de los controles de Flet utilizados en main.py

import flet as ft

def audit_controls():
    """
    # Esta función intenta instanciar cada control de Flet tal como se usa en la aplicación
    # para detectar errores de `TypeError` o `AttributeError` de forma proactiva.
    """
    print("Iniciando auditoría de controles de Flet...")

    try:
        # Auditoría de ft.Text
        ft.Text("Texto de prueba", size=16, weight="bold")
        print("ft.Text: OK")

        # Auditoría de ft.ProgressRing
        ft.ProgressRing(width=32, height=32, stroke_width=4)
        print("ft.ProgressRing: OK")

        # Auditoría de ft.TextField
        ft.TextField(label="Etiqueta", multiline=True, min_lines=4, max_lines=8, expand=True, disabled=True)
        print("ft.TextField: OK")

        # Auditoría de ft.Dropdown
        ft.Dropdown(label="Selecciona", options=[ft.dropdown.Option("opcion1")], disabled=True)
        print("ft.Dropdown: OK")

        # Auditoría de ft.IconButton
        ft.IconButton(icon="UNDO", tooltip="Tooltip", on_click=None, disabled=True)
        print("ft.IconButton: OK")

        # Auditoría de ft.ElevatedButton
        ft.ElevatedButton(content=ft.Text("Botón"), icon="SPEAKER_PHONE", on_click=None, disabled=True)
        print("ft.ElevatedButton: OK")

        # Auditoría de ft.ProgressBar
        ft.ProgressBar(value=0, width=100, visible=False)
        print("ft.ProgressBar: OK")

        # Auditoría de ft.Image (con el argumento 'fit' problemático)
        # Se usará una cadena de texto como valor para 'fit'
        ft.Image(src="", visible=False, width=600, height=300, fit="contain")
        print("ft.Image: OK")

        # Auditoría de ft.AlertDialog y ft.TextButton
        ft.AlertDialog(
            modal=True,
            title=ft.Text("Título"),
            content=ft.Text("Contenido"),
            actions=[ft.TextButton("OK")]
        )
        print("ft.AlertDialog y ft.TextButton: OK")

        # Auditoría de ft.FilePicker
        picker = ft.FilePicker()
        picker.on_result = None
        print("ft.FilePicker: OK")

        # Auditoría de ft.SnackBar
        ft.SnackBar(content=ft.Text("Mensaje"), bgcolor="green")
        print("ft.SnackBar: OK")

        # Auditoría de ft.Column y ft.Row con alineación
        # Se usarán cadenas de texto para las propiedades de alineación
        ft.Column(
            [ft.Text("item")],
            alignment="center",
            horizontal_alignment="center",
        )
        print("ft.Column: OK")

        ft.Row(
            [ft.Text("item")],
            alignment="center"
        )
        print("ft.Row: OK")

        print("\nAuditoría completada con éxito. Todos los controles parecen ser compatibles.")

    except Exception as e:
        print(f"\n¡Se ha encontrado un error de compatibilidad!")
        print(f"Error: {e}")
        print("Por favor, corrige el control correspondiente antes de continuar.")

if __name__ == "__main__":
    audit_controls()
