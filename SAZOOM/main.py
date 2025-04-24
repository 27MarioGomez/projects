import os, time
import requests
from typing import List, Optional, Dict
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY", "")

MEALDB_URL      = "https://www.themealdb.com/api/json/v1/1"
SPOONACULAR_URL = "https://api.spoonacular.com/recipes"
LIBRETRANSLATE  = "https://libretranslate.de/translate"
MYMEMORY        = "https://api.mymemory.translated.net/get"

app = FastAPI(title="Sazoom")
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)

# StaticFiles con cache control
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.middleware("http")
async def add_static_cache_control(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "public, max-age=86400"
    return response

def _lt(text: str, src: str, tgt: str) -> str:
    try:
        r = requests.post(
            LIBRETRANSLATE,
            json={"q": text, "source": src, "target": tgt, "format": "text"},
            timeout=3
        ).json()
        return r.get("translatedText", text)
    except:
        try:
            r = requests.get(
                MYMEMORY,
                params={"q": text, "langpair": f"{src}|{tgt}"},
                timeout=3
            ).json()
            return r.get("responseData", {}).get("translatedText", text)
        except:
            return text

def translate_text(text: str) -> str:
    return _lt(text, "en", "es")

def translate_list(items: List[str]) -> List[str]:
    return [translate_text(i) for i in items] if items else []

class RecipeSummary(BaseModel):
    id: str; title: str; image: str; source: str

class RecipeDetail(RecipeSummary):
    ingredients: List[str]
    instructions: List[str]
    time: Optional[int]
    servings: Optional[int]
    allergens: Optional[List[str]]
    nutrients: Optional[Dict[str,str]]

class SimpleCache:
    def __init__(self, ttl):
        self.ttl, self.store = ttl, {}
    def get(self, k):
        v = self.store.get(k)
        if not v or time.time() - v[0] > self.ttl:
            return None
        return v[1]
    def set(self, k, v):
        self.store[k] = (time.time(), v)

search_cache = SimpleCache(3600)
detail_cache = SimpleCache(86400)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/recipes", response_model=List[RecipeSummary])
async def search_recipes(
    ingredients: str = Query(...),
    diet: Optional[str] = None,
    intolerances: Optional[str] = None,
    number: int = Query(10, ge=1, le=30)
):
    ingr_en  = _lt(ingredients,  "es", "en")
    intol_en = _lt(intolerances, "es", "en") if intolerances else None

    key = f"{ingredients}|{diet}|{intolerances}|{number}"
    if cached := search_cache.get(key):
        return cached

    results: List[RecipeSummary] = []
    # TheMealDB
    m = requests.get(f"{MEALDB_URL}/filter.php",
                     params={"i": ingr_en}, timeout=5).json()
    if m.get("meals"):
        for meal in m["meals"]:
            results.append(RecipeSummary(
                id=f"mealdb_{meal['idMeal']}",
                title=meal["strMeal"],
                image=meal["strMealThumb"],
                source="mealdb"
            ))
    else:
        params = {
            "apiKey": SPOONACULAR_API_KEY,
            "includeIngredients": ingr_en,
            "number": number,
            "addRecipeInformation": True
        }
        if diet:      params["diet"]         = diet
        if intol_en:  params["intolerances"] = intol_en

        sp = requests.get(
            f"{SPOONACULAR_URL}/complexSearch",
            params=params, timeout=5
        ).json()
        for r in sp.get("results", []):
            results.append(RecipeSummary(
                id=f"spoon_{r['id']}",
                title=r["title"],
                image=r["image"],
                source="spoonacular"
            ))

    for r in results:
        r.title = translate_text(r.title)

    search_cache.set(key, results)
    return results

@app.get("/recipe/{recipe_id}")
async def recipe_page(
    request: Request,
    recipe_id: str,
    intolerances: Optional[str] = None
):
    detail = _fetch_detail(recipe_id, intolerances)
    return templates.TemplateResponse("detail.html", {
        "request": request, "recipe": detail
    })

@app.get("/recipes/{recipe_id}", response_model=RecipeDetail)
async def get_recipe(
    recipe_id: str,
    intolerances: Optional[str] = None
):
    return _fetch_detail(recipe_id, intolerances)

def _fetch_detail(recipe_id: str, intolerances: Optional[str]):
    key = f"{recipe_id}|{intolerances or ''}"
    if cached := detail_cache.get(key):
        return cached

    # (lógica TheMealDB vs Spoonacular, idéntica a la tuya…)

    detail.title        = translate_text(detail.title)
    detail.ingredients  = translate_list(detail.ingredients)
    detail.instructions = translate_list(detail.instructions)
    detail.allergens    = translate_list(detail.allergens or [])
    if detail.nutrients:
        keys    = list(detail.nutrients.keys())
        keys_es = translate_list(keys)
        detail.nutrients = {ke: detail.nutrients[k] for ke,k in zip(keys_es, keys)}

    detail_cache.set(key, detail)
    return detail

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0",
                port=int(os.getenv("PORT", 8000)), reload=False)
