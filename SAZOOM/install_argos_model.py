# install_argos_model.py
import argostranslate.package

# 1) Actualiza el índice de paquetes
argostranslate.package.update_package_index()

# 2) Obtén todos los paquetes disponibles
available_packages = argostranslate.package.get_available_packages()

# 3) Filtra el par EN→ES
en_es_pkg = next(
    pkg for pkg in available_packages
    if pkg.from_code == "en" and pkg.to_code == "es"
)

# 4) Descarga e instala
print(f"Descargando {en_es_pkg.name} versión {en_es_pkg.version}...")
download_path = en_es_pkg.download()
argostranslate.package.install_from_path(download_path)
print("✅ Modelo Argos EN→ES instalado correctamente.")
