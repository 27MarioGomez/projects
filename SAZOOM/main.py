import os, time, requests
from typing import List, Optional, Dict
from functools import lru_cache

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

import argostranslate.translate

# — Carga de env
load_dotenv()
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY", "")

MEALDB_URL      = "https://www.themealdb.com/api/json/v1/1"
SPOONACULAR_URL = "https://api.spoonacular.com/recipes"

# — Carga de traductores Argos offline
langs = argostranslate.translate.get_installed_languages()
lang_en = next(l for l in langs if l.code == "en")
lang_es = next(l for l in langs if l.code == "es")
trans_en_es = lang_en.get_translation(lang_es)
trans_es_en = lang_es.get_translation(lang_en)

@lru_cache(maxsize=2048)
def es2en(text: str) -> str:
    if not text: return ""
    try:    return trans_es_en.translate(text)
    except: return text

@lru_cache(maxsize=2048)
def en2es(text: str) -> str:
    if not text: return ""
    try:    return trans_en_es.translate(text)
    except: return text

def translate_list_en2es(items: List[str]) -> List[str]:
    return [en2es(i) for i in items] if items else []

# — App y middleware
app = FastAPI(title="Sazoom")
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.middleware("http")
async def add_static_cache_control(request: Request, call_next):
    resp = await call_next(request)
    if request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp

# — Modelos
class RecipeSummary(BaseModel):
    id: str; title: str; image: str; source: str

class RecipeDetail(RecipeSummary):
    ingredients: List[str]
    instructions: List[str]
    time: Optional[int]
    servings: Optional[int]
    allergens: Optional[List[str]]
    nutrients: Optional[Dict[str,str]]

# — Caché sencillo
class SimpleCache:
    def __init__(self, ttl): self.ttl, self.store = ttl, {}
    def get(self,k):
        v=self.store.get(k)
        return v[1] if v and time.time()-v[0]<self.ttl else None
    def set(self,k,v): self.store[k]=(time.time(),v)

search_cache = SimpleCache(3600)
detail_cache = SimpleCache(86400)

# — Rutas
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/recipes", response_model=List[RecipeSummary])
async def search_recipes(
    ingredients: str = Query(...), 
    diet: Optional[str] = None,
    intolerances: Optional[str] = None,
    number: int = Query(10,ge=1,le=30)
):
    # 1) Español→Inglés para API
    ingr_en  = es2en(ingredients)
    intol_en = es2en(intolerances) if intolerances else None

    key = f"{ingredients}|{diet}|{intolerances}|{number}"
    if cached := search_cache.get(key): return cached

    results: List[RecipeSummary] = []

    # MealDB
    m = requests.get(f"{MEALDB_URL}/filter.php",
                     params={"i":ingr_en},timeout=5).json()
    if m.get("meals"):
        for meal in m["meals"]:
            results.append(RecipeSummary(
                id=f"mealdb_{meal['idMeal']}",
                title=meal["strMeal"],
                image=meal["strMealThumb"],
                source="mealdb"
            ))
    else:
        # Spoonacular
        params = {
            "apiKey": SPOONACULAR_API_KEY,
            "includeIngredients":ingr_en,
            "number":number,
            "addRecipeInformation":True
        }
        if diet:     params["diet"]=diet
        if intol_en: params["intolerances"]=intol_en

        sp = requests.get(
            f"{SPOONACULAR_URL}/complexSearch",
            params=params, timeout=5
        ).json()
        for r in sp.get("results",[]):
            results.append(RecipeSummary(
                id=f"spoon_{r['id']}",
                title=r["title"],
                image=r["image"],
                source="spoonacular"
            ))

    # 2) Inglés→Español títulos
    for r in results:
        r.title = en2es(r.title)

    search_cache.set(key, results)
    return results

@app.get("/recipe/{recipe_id}")
async def recipe_page(request: Request, recipe_id: str,
                      intolerances: Optional[str]=None):
    try:
        detail = _fetch_detail(recipe_id, intolerances)
        return templates.TemplateResponse("detail.html", {
            "request": request, "recipe": detail
        })
    except HTTPException as he:
        return templates.TemplateResponse("error.html",
                                          {"request":request},
                                          status_code=he.status_code)
    except Exception:
        import traceback; traceback.print_exc()
        return templates.TemplateResponse("error.html",
                                          {"request":request},
                                          status_code=500)

@app.get("/recipes/{recipe_id}", response_model=RecipeDetail)
async def get_recipe(recipe_id: str,
                     intolerances: Optional[str]=None):
    return _fetch_detail(recipe_id, intolerances)

def _fetch_detail(recipe_id: str, intolerances: Optional[str]):
    key = f"{recipe_id}|{intolerances or ''}"
    if cached := detail_cache.get(key): return cached

    # TheMealDB
    if recipe_id.startswith("mealdb_"):
        mid = recipe_id.split("_",1)[1]
        data = requests.get(
            f"{MEALDB_URL}/lookup.php",
            params={"i":mid}, timeout=5
        ).json().get("meals",[{}])[0]
        if not data: raise HTTPException(404,"No encontrada")

        ing,alg = [],[]
        for i in range(1,21):
            nm = data.get(f"strIngredient{i}")
            ms = data.get(f"strMeasure{i}")
            if nm:
                ing.append(f"{ms.strip()} {nm.strip()}")
                if any(a in nm.lower() for a in
                       ["milk","egg","peanut","nut","wheat","soy"]):
                    alg.append(nm)
        instr = [s.strip() for s in
                 data.get("strInstructions","").split(".") if s.strip()]

        detail = RecipeDetail(
            id=recipe_id, title=data["strMeal"],
            image=data["strMealThumb"], source="mealdb",
            ingredients=ing, instructions=instr,
            time=None, servings=None,
            allergens=alg, nutrients=None
        )

    # Spoonacular
    elif recipe_id.startswith("spoon_"):
        sid = recipe_id.split("_",1)[1]
        d = requests.get(
            f"{SPOONACULAR_URL}/{sid}/information",
            params={"apiKey":SPOONACULAR_API_KEY,
                    "includeNutrition":True},
            timeout=5
        ).json()
        if not d.get("id"): raise HTTPException(404,"No encontrada")

        ing = [f"{i['amount']} {i['unit']} {i['name']}"
               for i in d.get("extendedIngredients",[])]
        instr=[]
        for blk in d.get("analyzedInstructions",[]):
            instr += [st["step"] for st in blk.get("steps",[])]
        nutr = {
            n["name"]:f"{n['amount']}{n['unit']}"
            for n in d.get("nutrition",{}).get("nutrients",[])
            if n["name"].lower() in
               ("calories","fat","carbohydrates","protein")
        }

        detail = RecipeDetail(
            id=recipe_id, title=d["title"],
            image=d["image"], source="spoonacular",
            ingredients=ing, instructions=instr,
            time=d.get("readyInMinutes"),
            servings=d.get("servings"),
            allergens=(intolerances.split(",")
                       if intolerances else []),
            nutrients=nutr
        )
    else:
        raise HTTPException(400,"ID inválido")

    # 3) Traducción EN→ES de detalle
    detail.title        = en2es(detail.title)
    detail.ingredients  = translate_list_en2es(detail.ingredients)
    detail.instructions = translate_list_en2es(detail.instructions)
    detail.allergens    = translate_list_en2es(detail.allergens or [])
    if detail.nutrients:
        ks = list(detail.nutrients.keys())
        ks_es = translate_list_en2es(ks)
        detail.nutrients = {kes: detail.nutrients[k]
                            for kes,k in zip(ks_es, ks)}

    detail_cache.set(key, detail)
    return detail

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0",
                port=int(os.getenv("PORT",8000)),
                reload=False)
