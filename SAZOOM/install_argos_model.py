import argostranslate.package
import argostranslate.translate
import tempfile
import urllib.request
import os

# URL oficial del paquete EN→ES
url = "https://www.argosopentech.com/argospm/packages/translate-en_es.argosmodel"
# Guarda en un directorio temporal
path = os.path.join(tempfile.gettempdir(), "translate-en_es.argosmodel")

print(f"Descargando modelo a {path}...")
urllib.request.urlretrieve(url, path)
print("Instalando modelo Argos EN→ES...")
argostranslate.package.install_from_path(path)
print("✅ Modelo Argos EN→ES instalado correctamente.")
