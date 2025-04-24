# install_argos_model.py

import argostranslate.package

# 1) Actualiza el índice de paquetes
argostranslate.package.update_package_index()

# 2) Obtén todos los paquetes disponibles
available_packages = argostranslate.package.get_available_packages()

# 3) Filtra e instala tanto EN→ES como ES→EN
for pkg in available_packages:
    if (pkg.from_code, pkg.to_code) in [("en", "es"), ("es", "en")]:
        print(f"Descargando modelo {pkg.from_code} → {pkg.to_code} v{pkg.package_version}…")
        path = pkg.download()
        argostranslate.package.install_from_path(path)
        print(f"✅ Modelo {pkg.from_code} → {pkg.to_code} instalado correctamente.")
