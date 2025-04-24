import os
import time
import requests
from typing import List, Optional, Dict
from functools import wraps

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

# — Carga de .env
load_dotenv()
EDAMAM_APP_ID  = os.getenv("EDAMAM_APP_ID", "")
EDAMAM_APP_KEY = os.getenv("EDAMAM_APP_KEY", "")

if not EDAMAM_APP_ID or not EDAMAM_APP_KEY:
    raise RuntimeError("Define EDAMAM_APP_ID y EDAMAM_APP_KEY en .env")

# — Endpoint Spanish Beta
BASE_SEARCH_URL = "https://test-es.edamam.com/search"  # Edamam Spanish Beta :contentReference[oaicite:3]{index=3}

# — FastAPI setup
app = FastAPI(title="Sazoom")
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# — Decorador TTL cache
def ttl_cache(ttl: int):
    def decorator(func):
        cache = {}
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            entry = cache.get(key)
            if entry and time.time() - entry[0] < ttl:
                return entry[1]
            result = func(*args, **kwargs)
            cache[key] = (time.time(), result)
            return result
        return wrapper
    return decorator

# — Modelos Pydantic
class RecipeSummary(BaseModel):
    id: str
    title: str
    image: str
    source: str = "edamam"

class RecipeDetail(RecipeSummary):
    ingredients: List[str]
    instructions_url: str
    diet: List[str]
    health: List[str]
    nutrients: Dict[str, str]

# — Rutas
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/recipes", response_model=List[RecipeSummary])
@ttl_cache(ttl=3600)  # Cache 1h
def search_recipes(
    ingredients: str = Query(..., description="Ingredientes en español, separados por comas"),
    diet: Optional[str]   = Query(None, description="Dieta (en inglés, p.ej. keto, vegan)"),
    health: Optional[str] = Query(None, description="Etiquetas health/allergy en inglés"),
    number: int           = Query(10, ge=1, le=30)
):
    params = {
        "q":       ingredients,
        "app_id":  EDAMAM_APP_ID,
        "app_key": EDAMAM_APP_KEY,
        "limit":   number
    }
    if diet:
        params["diet"] = diet
    if health:
        params["health"] = health

    resp = requests.get(BASE_SEARCH_URL, params=params, timeout=5)
    # 1) Validar estado HTTP :contentReference[oaicite:4]{index=4}
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, f"Error Edamam: {resp.text}")
    # 2) Validar JSON :contentReference[oaicite:5]{index=5}
    content_type = resp.headers.get("Content-Type", "")
    if "application/json" not in content_type:
        raise HTTPException(502, "Edamam no devolvió JSON válido")
    # 3) Parseo seguro :contentReference[oaicite:6]{index=6}
    try:
        hits = resp.json().get("hits", [])
    except ValueError:
        raise HTTPException(502, "JSON malformado recibido de Edamam")

    results: List[RecipeSummary] = []
    for hit in hits:
        r = hit.get("recipe", {})
        # Extrae id de la URI
        uri_id = r.get("uri","").split("#")[-1]
        results.append(RecipeSummary(
            id=uri_id,
            title=r.get("label",""),
            image=r.get("image","")
        ))
    return results

@app.get("/recipe/{recipe_id}", response_model=RecipeDetail)
@ttl_cache(ttl=86400)  # Cache 24h
def get_recipe(recipe_id: str):
    url = f"{BASE_SEARCH_URL}/{recipe_id}"
    params = {"type":"public", "app_id":EDAMAM_APP_ID, "app_key":EDAMAM_APP_KEY}
    resp = requests.get(url, params=params, timeout=5)

    if resp.status_code == 404:
        raise HTTPException(404, "Receta no encontrada")
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, f"Error Edamam: {resp.text}")
    if "application/json" not in resp.headers.get("Content-Type",""):
        raise HTTPException(502, "Edamam no devolvió JSON válido")

    try:
        data = resp.json().get("recipe", {})
    except ValueError:
        raise HTTPException(502, "JSON malformado recibido de Edamam")

    # Mapa de nutrientes de interés
    nut_map = {
        "ENERC_KCAL": "Calorías",
        "FAT":        "Grasas",
        "CHOCDF":     "Carbohidratos",
        "PROCNT":     "Proteínas"
    }
    nutrients = {}
    for key, info in data.get("totalNutrients", {}).items():
        if key in nut_map and info.get("quantity") is not None:
            nutrients[nut_map[key]] = f"{info['quantity']:.1f}{info['unit']}"

    detail = RecipeDetail(
        id=recipe_id,
        title=data.get("label",""),
        image=data.get("image",""),
        ingredients=data.get("ingredientLines", []),
        instructions_url=data.get("url",""),
        diet=data.get("dietLabels", []),
        health=data.get("healthLabels", []),
        nutrients=nutrients
    )
    return detail

@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return templates.TemplateResponse(
        "error.html",
        {"request": request, "detail": exc.detail},
        status_code=exc.status_code
    )
