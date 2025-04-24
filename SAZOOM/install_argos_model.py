import argostranslate.package
import argostranslate.translate

# 1) Actualiza el índice local de paquetes
argostranslate.package.update_package_index()

# 2) Busca el paquete EN→ES
available_packages = argostranslate.package.get_available_packages()
en_es_pkg = next(
    pkg for pkg in available_packages
    if pkg.from_code == "en" and pkg.to_code == "es"
)

# 3) Descarga e instala
print(f"Descargando {en_es_pkg.package_name} versión {en_es_pkg.package_version}...")
download_path = en_es_pkg.download()
argostranslate.package.install_from_path(download_path)
print("✅ Modelo Argos EN→ES instalado correctamente.")
