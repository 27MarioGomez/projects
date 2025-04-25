import os
import time
import requests
from typing import List, Dict, Any, Tuple, Optional
from functools import wraps

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

# — Carga de .env
load_dotenv()
SPOON_KEY = os.getenv("SPOONACULAR_API_KEY", "")
if not SPOON_KEY:
    raise RuntimeError("Define SPOONACULAR_API_KEY en .env")

# — Endpoints
MEALDB_FILTER_URL     = "https://www.themealdb.com/api/json/v1/1/filter.php"
MEALDB_LOOKUP_URL     = "https://www.themealdb.com/api/json/v1/1/lookup.php"
SPOON_SEARCH_URL      = "https://api.spoonacular.com/recipes/complexSearch"
SPOON_INFO_URL        = "https://api.spoonacular.com/recipes/{id}/information"

# — App
app = FastAPI(title="Sazoom")
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# — Caché TTL genérico
def ttl_cache(ttl: int):
    def deco(fn):
        cache: Dict[Tuple, Any] = {}
        @wraps(fn)
        def wr(*args, **kwargs):
            # normalizar args/kwargs a hashable
            norm_args = tuple(tuple(a) if isinstance(a, list) else a for a in args)
            norm_kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in kwargs.items()}
            key = (fn.__name__, norm_args, frozenset(norm_kwargs.items()))
            entry = cache.get(key)
            if entry and time.time() - entry[0] < ttl:
                return entry[1]
            res = fn(*args, **kwargs)
            cache[key] = (time.time(), res)
            return res
        return wr
    return deco

# — Modelos
class RecipeSummary(BaseModel):
    id: str
    title: str
    image: str
    source: str

class RecipeDetail(BaseModel):
    id: str
    title: str
    image: str
    ingredients: List[str]
    instructions: str
    source: str

# — Home con listas para dropdowns
@app.get("/")
async def home(request: Request):
    all_ingredients = [
      "chicken","beef","pork","fish","shrimp","rice","potato","tomato",
      "onion","garlic","pepper","carrot","egg","milk","cheese","butter",
      "flour","sugar","salt","olive oil","lemon","spinach","mushroom",
      "broccoli","corn","beans","yogurt","tuna","bread"
    ]
    all_diets = ["balanced","high-protein","low-fat","low-carb"]
    all_health = ["gluten-free","dairy-free","peanut-free","vegan","vegetarian"]
    return templates.TemplateResponse("index.html", {
      "request": request,
      "all_ingredients": all_ingredients,
      "all_diets": all_diets,
      "all_health": all_health
    })

# — Carga IDs de MealDB para un ingrediente
@ttl_cache(ttl=3600)
def fetch_mealdb_ids(ing: str) -> set:
    r = requests.get(MEALDB_FILTER_URL, params={"i": ing}, timeout=5)
    r.raise_for_status()
    return {m["idMeal"] for m in (r.json().get("meals") or [])}

# — Detalle MealDB
@ttl_cache(ttl=86400)
def fetch_mealdb_detail(mid: str) -> Dict[str, Any]:
    r = requests.get(MEALDB_LOOKUP_URL, params={"i": mid}, timeout=5)
    r.raise_for_status()
    meals = r.json().get("meals") or []
    return meals[0] if meals else {}

# — Detalle Spoonacular
@ttl_cache(ttl=86400)
def fetch_spoon_detail(sid: str) -> Dict[str, Any]:
    url = SPOON_INFO_URL.format(id=sid)
    r = requests.get(url, params={"apiKey": SPOON_KEY}, timeout=5)
    r.raise_for_status()
    return r.json()

# — Búsqueda combinada
@app.get("/recipes", response_model=List[RecipeSummary])
@ttl_cache(ttl=3600)
def search_recipes(
    ingredients: List[str] = Query(..., description="Selecciona ingredientes"),
    diet: Optional[str]   = Query(None, description="Filtro dieta"),
    health: List[str]     = Query([], description="Filtros health/alérgenos"),
    number: int           = Query(12, ge=1, le=50)
):
    sel = {i.lower() for i in ingredients}

    # 1) MealDB: intersección
    id_sets = [fetch_mealdb_ids(ing) for ing in sel]
    common = set.intersection(*id_sets) if id_sets else set()

    results: List[RecipeSummary] = []
    # filtro subset
    for mid in list(common)[:number]:
        meal = fetch_mealdb_detail(mid)
        ingr_list = []
        for i in range(1, 21):
            ing = meal.get(f"strIngredient{i}") or ""
            meas = meal.get(f"strMeasure{i}")   or ""
            if ing.strip():
                ingr_list.append(f"{meas.strip()} {ing.strip()}")

        
        if set(ingr_list).issubset(sel):
            results.append(RecipeSummary(
                id=mid, title=meal["strMeal"],
                image=meal["strMealThumb"],
                source="mealdb"
            ))

    # 2) Spoonacular (complemento)
    if len(results) < number:
        need = number - len(results)
        params = {
            "apiKey": SPOON_KEY,
            "includeIngredients": ",".join(ingredients),
            "number": need,
            "addRecipeInformation": False
        }
        if diet:   params["diet"] = diet
        if health: params["intolerances"] = ",".join(health)

        data = requests.get(SPOON_SEARCH_URL, params=params, timeout=5).json()
        for r in data.get("results", []):
            results.append(RecipeSummary(
                id=str(r["id"]), title=r["title"],
                image=r["image"], source="spoonacular"
            ))

    return results

# — Detalle combinado
@app.get("/recipe/{recipe_id}")
async def recipe_page(request: Request, recipe_id: str):
    # prueba MealDB
    meal = fetch_mealdb_detail(recipe_id)
    if meal and meal.get("idMeal"):
        ingredients = [
            f"{meal.get(f'strMeasure{i}','').strip()} {meal.get(f'strIngredient{i}','').strip()}"
            for i in range(1,21)
            if meal.get(f'strIngredient{i}','').strip()
        ]
        detail = RecipeDetail(
            id=recipe_id, title=meal["strMeal"],
            image=meal["strMealThumb"],
            ingredients=ingredients,
            instructions=meal.get("strInstructions",""),
            source="mealdb"
        )
    else:
        info = fetch_spoon_detail(recipe_id)
        ingredients = [
            f"{ing['amount']} {ing['unit']} {ing['name']}"
            for ing in info.get("extendedIngredients",[])
        ]
        detail = RecipeDetail(
            id=recipe_id, title=info.get("title",""),
            image=info.get("image",""),
            ingredients=ingredients,
            instructions=info.get("instructions",""),
            source="spoonacular"
        )

    return templates.TemplateResponse("detail.html", {
        "request": request, "recipe": detail
    })

# — Error handler
@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return templates.TemplateResponse(
        "error.html",
        {"request": request, "detail": exc.detail},
        status_code=exc.status_code
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
