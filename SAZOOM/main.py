import os
import time
import re
import requests
from typing import List, Optional, Dict
from functools import lru_cache

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

import argostranslate.translate

# — Cargar variables de entorno
load_dotenv()
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY", "")

MEALDB_URL      = "https://www.themealdb.com/api/json/v1/1"
SPOONACULAR_URL = "https://api.spoonacular.com/recipes"

# — Inicializar traductores Argos offline
langs     = argostranslate.translate.get_installed_languages()
lang_en   = next(l for l in langs if l.code == "en")
lang_es   = next(l for l in langs if l.code == "es")
trans_es_en = lang_es.get_translation(lang_en)
trans_en_es = lang_en.get_translation(lang_es)

@lru_cache(maxsize=2048)
def es2en(text: str) -> str:
    return trans_es_en.translate(text) if text else ""

@lru_cache(maxsize=2048)
def en2es(text: str) -> str:
    return trans_en_es.translate(text) if text else ""

def normalize_units(text: str) -> str:
    patterns = {
        r'\b(\d+\.?\d*)\s?[Tt]\b': r'\1 cucharada',
        r'\b(\d+\.?\d*)\s?[t]\b':  r'\1 cucharadita',
        r'\b(\d+\.?\d*)\s?[Oo][Zz]\b': r'\1 onza',
        r'\b(\d+\.?\d*)\s?[Gg]\b':  r'\1 gramos',
        r'\b(\d+\.?\d*)\s?[Mm][Ll]\b': r'\1 ml',
    }
    for pat, repl in patterns.items():
        text = re.sub(pat, repl, text)
    return text

def translate_and_normalize(text: str) -> str:
    return normalize_units(en2es(text))

def translate_list(items: List[str]) -> List[str]:
    return [translate_and_normalize(i) for i in items]

# — FastAPI setup
app = FastAPI(title="Sazoom")
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# — Simple in-memory cache
class SimpleCache:
    def __init__(self, ttl): self.ttl, self.store = ttl, {}
    def get(self, k):
        v = self.store.get(k)
        return v[1] if v and time.time() - v[0] < self.ttl else None
    def set(self, k, v): self.store[k] = (time.time(), v)

search_cache = SimpleCache(3600)
detail_cache = SimpleCache(86400)

# — Home page
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# — API: search recipes
class RecipeSummary(BaseModel):
    id: str
    title: str
    image: str
    source: str

@app.get("/recipes", response_model=List[RecipeSummary])
async def search_recipes(
    ingredients: str = Query(...),
    diet: Optional[str] = None,
    intolerances: Optional[str] = None,
    number: int = Query(10, ge=1, le=30)
):
    ingr_en  = es2en(ingredients)
    intol_en = es2en(intolerances) if intolerances else None
    key = f"{ingredients}|{diet}|{intolerances}|{number}"
    if cached := search_cache.get(key):
        return cached

    results: List[RecipeSummary] = []

    # 1) TheMealDB
    m = requests.get(
        f"{MEALDB_URL}/filter.php",
        params={"i": ingr_en}, timeout=5
    ).json()
    if m.get("meals"):
        for meal in m["meals"]:
            results.append(RecipeSummary(
                id=f"mealdb_{meal['idMeal']}",
                title=translate_and_normalize(meal["strMeal"]),
                image=meal["strMealThumb"],
                source="mealdb"
            ))
    else:
        # 2) Spoonacular
        params = {
            "apiKey": SPOONACULAR_API_KEY,
            "includeIngredients": ingr_en,
            "number": number,
            "addRecipeInformation": True
        }
        if diet:      params["diet"] = diet
        if intol_en:  params["intolerances"] = intol_en

        sp = requests.get(
            f"{SPOONACULAR_URL}/complexSearch",
            params=params, timeout=5
        ).json()
        for r in sp.get("results", []):
            results.append(RecipeSummary(
                id=f"spoon_{r['id']}",
                title=translate_and_normalize(r["title"]),
                image=r["image"],
                source="spoonacular"
            ))

    search_cache.set(key, results)
    return results

# — Recipe detail page
class RecipeDetail(RecipeSummary):
    ingredients: List[str]
    instructions: List[str]
    time: Optional[int]
    servings: Optional[int]
    allergens: Optional[List[str]]
    nutrients: Optional[Dict[str, str]]

@app.get("/recipe/{recipe_id}")
async def recipe_page(request: Request, recipe_id: str,
                      intolerances: Optional[str] = None):
    try:
        detail = _fetch_detail(recipe_id, intolerances)
        return templates.TemplateResponse("detail.html", {
            "request": request, "recipe": detail
        })
    except HTTPException as he:
        return templates.TemplateResponse(
            "error.html", {"request": request},
            status_code=he.status_code
        )
    except Exception:
        return templates.TemplateResponse(
            "error.html", {"request": request}, status_code=500
        )

@app.get("/recipes/{recipe_id}", response_model=RecipeDetail)
async def get_recipe(recipe_id: str,
                     intolerances: Optional[str] = None):
    return _fetch_detail(recipe_id, intolerances)

def _fetch_detail(recipe_id: str, intolerances: Optional[str]):
    key = f"{recipe_id}|{intolerances or ''}"
    if cached := detail_cache.get(key):
        return cached

    # TheMealDB detail
    if recipe_id.startswith("mealdb_"):
        mid = recipe_id.split("_", 1)[1]
        data = requests.get(
            f"{MEALDB_URL}/lookup.php",
            params={"i": mid}, timeout=5
        ).json().get("meals", [{}])[0]
        if not data:
            raise HTTPException(404, "Receta no encontrada")

        ing, alg = [], []
        for i in range(1, 21):
            nm = data.get(f"strIngredient{i}")
            ms = data.get(f"strMeasure{i}")
            if nm:
                txt = f"{ms.strip()} {nm.strip()}"
                ing.append(txt)
                if any(a in nm.lower() for a in
                       ["milk", "egg", "peanut", "nut", "wheat", "soy"]):
                    alg.append(nm)
        instr = [s.strip() for s in
                 data.get("strInstructions", "").split(".") if s.strip()]

        detail = RecipeDetail(
            id=recipe_id,
            title=translate_and_normalize(data["strMeal"]),
            image=data["strMealThumb"],
            source="mealdb",
            ingredients=translate_list(ing),
            instructions=translate_list(instr),
            time=None,
            servings=None,
            allergens=translate_list(alg),
            nutrients=None
        )

    # Spoonacular detail
    elif recipe_id.startswith("spoon_"):
        sid = recipe_id.split("_", 1)[1]
        d = requests.get(
            f"{SPOONACULAR_URL}/{sid}/information",
            params={"apiKey": SPOONACULAR_API_KEY,
                    "includeNutrition": True},
            timeout=5
        ).json()
        if not d.get("id"):
            raise HTTPException(404, "Receta no encontrada")

        ing = [f"{i['amount']} {i['unit']} {i['name']}"
               for i in d.get("extendedIngredients", [])]
        instr = []
        for blk in d.get("analyzedInstructions", []):
            instr += [st["step"] for st in blk.get("steps", [])]
        nutr = {
            n["name"]: f"{n['amount']}{n['unit']}"
            for n in d.get("nutrition", {}).get("nutrients", [])
            if n["name"].lower() in
               ("calories", "fat", "carbohydrates", "protein")
        }

        detail = RecipeDetail(
            id=recipe_id,
            title=translate_and_normalize(d["title"]),
            image=d["image"],
            source="spoonacular",
            ingredients=translate_list(ing),
            instructions=translate_list(instr),
            time=d.get("readyInMinutes"),
            servings=d.get("servings"),
            allergens=translate_list(
                intolerances.split(",") if intolerances else []
            ),
            nutrients={translate_and_normalize(k): v
                       for k, v in nutr.items()}
        )
    else:
        raise HTTPException(400, "ID inválido")

    detail_cache.set(key, detail)
    return detail

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False
    )
