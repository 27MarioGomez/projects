import os
import time
import requests
from typing import List, Optional, Dict

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from functools import wraps

# — Carga de variables de entorno
load_dotenv()
EDAMAM_APP_ID = os.getenv("EDAMAM_APP_ID", "")
EDAMAM_APP_KEY = os.getenv("EDAMAM_APP_KEY", "")
BASE_URL = "https://test-es.edamam.com/api/recipes/v2"

if not EDAMAM_APP_ID or not EDAMAM_APP_KEY:
    raise RuntimeError("Define EDAMAM_APP_ID y EDAMAM_APP_KEY en .env")

# — App y middleware
app = FastAPI(title="Sazoom")
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# — Simple TTL cache decorator
def ttl_cache(ttl: int):
    def decorator(fn):
        cache = {}
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            entry = cache.get(key)
            if entry and time.time() - entry[0] < ttl:
                return entry[1]
            result = fn(*args, **kwargs)
            cache[key] = (time.time(), result)
            return result
        return wrapper
    return decorator

# — Modelos
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
@ttl_cache(ttl=3600)
def search_recipes(
    ingredients: str = Query(..., description="Ingredientes en español, separados por comas"),
    diet: Optional[str] = Query(None, description="Dieta (keto, vegan, etc.) en inglés"),
    health: Optional[str] = Query(None, description="Alergenos/intolerancias en inglés, separados por comas"),
    number: int = Query(10, ge=1, le=30)
):
    params = {
        "type": "public",
        "q": ingredients,
        "app_id": EDAMAM_APP_ID,
        "app_key": EDAMAM_APP_KEY,
        "imageSize": "REGULAR",
        "field": ["uri","label","image"],
        "limit": number
    }
    if diet:
        params["diet"] = diet
    if health:
        params["health"] = health

    resp = requests.get(BASE_URL, params=params, timeout=5)
    resp.raise_for_status()
    data = resp.json().get("hits", [])
    results = []
    for hit in data:
        r = hit["recipe"]
        # extraer id interno de URI
        uri_id = r["uri"].split("#")[-1]
        results.append(RecipeSummary(
            id=uri_id,
            title=r["label"],
            image=r["image"],
        ))
    return results

@app.get("/recipe/{recipe_id}")
@ttl_cache(ttl=86400)
def get_recipe(recipe_id: str):
    # Detalles completos
    url = f"{BASE_URL}/{recipe_id}"
    params = {
        "type": "public",
        "app_id": EDAMAM_APP_ID,
        "app_key": EDAMAM_APP_KEY
    }
    resp = requests.get(url, params=params, timeout=5)
    if resp.status_code == 404:
        raise HTTPException(404, "Receta no encontrada")
    resp.raise_for_status()
    r = resp.json().get("recipe", {})

    # Nutrientes de interés
    nutrients = {}
    for key, info in r.get("totalNutrients", {}).items():
        if info.get("unit") and info.get("quantity"):
            # coger solo kcal, FAT, CHOCDF, PROCNT
            if key in ("ENERC_KCAL", "FAT", "CHOCDF", "PROCNT"):
                label = {
                    "ENERC_KCAL": "Calorías",
                    "FAT": "Grasas",
                    "CHOCDF": "Carbohidratos",
                    "PROCNT": "Proteínas"
                }[key]
                nutrients[label] = f"{info['quantity']:.1f}{info['unit']}"

    detail = RecipeDetail(
        id=recipe_id,
        title=r.get("label",""),
        image=r.get("image",""),
        ingredients=r.get("ingredientLines", []),
        instructions_url=r.get("url",""),
        diet=r.get("dietLabels", []),
        health=r.get("healthLabels", []),
        nutrients=nutrients
    )
    return detail

@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return templates.TemplateResponse(
        "error.html",
        {"request": request, "status_code": exc.status_code, "detail": exc.detail},
        status_code=exc.status_code
    )
