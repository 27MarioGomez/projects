import os, time, requests
from typing import List, Optional, Dict
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from deep_translator import GoogleTranslator

load_dotenv()
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY", "")

MEALDB_URL = "https://www.themealdb.com/api/json/v1/1"
SPOONACULAR_URL = "https://api.spoonacular.com/recipes"

app = FastAPI(title="Sazoom")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Instancias de traductor
es_en = GoogleTranslator(source='es', target='en')
en_es = GoogleTranslator(source='en', target='es')

def batch_traduce(traductor: GoogleTranslator, texto_comas: str) -> str:
    if not texto_comas:
        return ""
    items = [i.strip() for i in texto_comas.split(",") if i.strip()]
    traducidos = traductor.translate_batch(items)
    return ",".join(traducidos)

def lista_traduce(traductor: GoogleTranslator, items: List[str]) -> List[str]:
    if not items:
        return []
    return traductor.translate_batch(items)

class RecipeSummary(BaseModel):
    id: str
    title: str
    image: str
    source: str

class RecipeDetail(RecipeSummary):
    ingredients: List[str]
    instructions: List[str]
    time: Optional[int]
    servings: Optional[int]
    allergens: Optional[List[str]]
    nutrients: Optional[Dict[str, str]]

class SimpleCache:
    def __init__(self, ttl): self.ttl, self.store = ttl, {}
    def get(self, k):
        v = self.store.get(k)
        if not v or time.time() - v[0] > self.ttl: return None
        return v[1]
    def set(self, k, v):
        self.store[k] = (time.time(), v)

search_cache = SimpleCache(3600)
detail_cache = SimpleCache(86400)

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/recipes", response_model=List[RecipeSummary])
def search_recipes(
    ingredients: str = Query(...),
    diet: Optional[str] = None,
    intolerances: Optional[str] = None,
    number: int = Query(10, ge=1, le=30)
):
    # traduce los inputs ES→EN
    ingr_en = batch_traduce(es_en, ingredients)
    diet_en = diet  # dieta usamos misma palabra en inglés si el usuario la mete en inglés
    intol_en = batch_traduce(es_en, intolerances) if intolerances else None

    cache_key = f"{ingredients}|{diet}|{intolerances}|{number}"
    if cached := search_cache.get(cache_key):
        return cached

    results: List[RecipeSummary] = []
    # 1) TheMealDB
    resp = requests.get(f"{MEALDB_URL}/filter.php", params={"i":ingr_en}, timeout=5).json()
    if resp.get("meals"):
        for m in resp["meals"]:
            results.append(RecipeSummary(
                id=f"mealdb_{m['idMeal']}",
                title=m["strMeal"],
                image=m["strMealThumb"],
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
        if diet_en: params["diet"] = diet_en
        if intol_en: params["intolerances"] = intol_en
        sp = requests.get(f"{SPOONACULAR_URL}/complexSearch", params=params, timeout=5).json()
        for r in sp.get("results", []):
            results.append(RecipeSummary(
                id=f"spoon_{r['id']}",
                title=r["title"],
                image=r["image"],
                source="spoonacular"
            ))

    # traduce títulos EN→ES
    titles = [r.title for r in results]
    titles_es = lista_traduce(en_es, titles)
    for r, te in zip(results, titles_es):
        r.title = te

    search_cache.set(cache_key, results)
    return results

@app.get("/recipes/{recipe_id}", response_model=RecipeDetail)
def get_recipe(recipe_id: str, intolerances: Optional[str] = None):
    cache_key = f"{recipe_id}|{intolerances or ''}"
    if cached := detail_cache.get(cache_key):
        return cached

    # obtiene datos
    if recipe_id.startswith("mealdb_"):
        mid = recipe_id.split("_",1)[1]
        data = requests.get(f"{MEALDB_URL}/lookup.php", params={"i":mid}, timeout=5).json().get("meals",[{}])[0]
        if not data: raise HTTPException(404,"Not found")
        ing, alg = [], []
        for i in range(1,21):
            nm = data.get(f"strIngredient{i}")
            ms = data.get(f"strMeasure{i}")
            if nm:
                ing.append(f"{ms.strip()} {nm.strip()}")
                if any(a in nm.lower() for a in ["milk","egg","peanut","nut","wheat","soy"]):
                    alg.append(nm)
        instr = [s.strip() for s in data.get("strInstructions","").split(".") if s.strip()]
        detail = RecipeDetail(
            id=recipe_id,
            title=data["strMeal"],
            image=data["strMealThumb"],
            source="mealdb",
            ingredients=ing,
            instructions=instr,
            time=None, servings=None,
            allergens=alg,
            nutrients=None
        )
    elif recipe_id.startswith("spoon_"):
        sid = recipe_id.split("_",1)[1]
        d = requests.get(f"{SPOONACULAR_URL}/{sid}/information",
                         params={"apiKey":SPOONACULAR_API_KEY,"includeNutrition":True},
                         timeout=5).json()
        if not d.get("id"): raise HTTPException(404,"Not found")
        ing = [f"{i['amount']} {i['unit']} {i['name']}" for i in d.get("extendedIngredients",[])]
        instr = []
        for blk in d.get("analyzedInstructions",[]):
            instr += [s["step"] for s in blk.get("steps",[])]
        nutr = {
            n["name"]:f"{n['amount']}{n['unit']}"
            for n in d.get("nutrition",{}).get("nutrients",[])
            if n["name"].lower() in ("calories","fat","carbohydrates","protein")
        }
        detail = RecipeDetail(
            id=recipe_id,
            title=d["title"],
            image=d["image"],
            source="spoonacular",
            ingredients=ing,
            instructions=instr,
            time=d.get("readyInMinutes"),
            servings=d.get("servings"),
            allergens=(intolerances.split(",") if intolerances else []),
            nutrients=nutr
        )
    else:
        raise HTTPException(400,"Invalid ID")

    # traduce todo EN→ES
    detail.title = en_es.translate(detail.title)
    detail.ingredients = lista_traduce(en_es, detail.ingredients)
    detail.instructions = lista_traduce(en_es, detail.instructions)
    detail.allergens = lista_traduce(en_es, detail.allergens or [])
    if detail.nutrients:
        keys = list(detail.nutrients.keys())
        keys_es = lista_traduce(en_es, keys)
        detail.nutrients = {ke: detail.nutrients[k] for ke,k in zip(keys_es, keys)}

    detail_cache.set(cache_key, detail)
    return detail

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0",
                port=int(os.getenv("PORT",8000)), reload=True)
