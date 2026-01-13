import flet as ft

# Lista todos los atributos del módulo de iconos
all_icons = dir(ft.icons)

# Filtra para quedarnos solo con los nombres de los iconos (que suelen ser en mayúsculas)
icon_names = [icon for icon in all_icons if icon.isupper()]

# Imprime la lista de nombres de iconos
for name in sorted(icon_names):
    print(name)
