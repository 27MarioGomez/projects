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

# Carpeta estática y plantillas
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Modelos
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
    nutrients: Optional[Dict[str,str]]

# Caché con TTL
class SimpleCache:
    def __init__(self, ttl):
        self.ttl = ttl
        self.store = {}
    def get(self, k):
        v = self.store.get(k)
        if not v or time.time() - v[0] > self.ttl:
            return None
        return v[1]
    def set(self, k, v):
        self.store[k] = (time.time(), v)

search_cache = SimpleCache(3600)   # 1h
detail_cache = SimpleCache(86400)  # 24h

# Función de traducción
def traducir_ingredientes(texto_es: str) -> str:
    lista = [i.strip() for i in texto_es.split(",") if i.strip()]
    traducidos = GoogleTranslator(source='es', target='en').translate_batch(lista)
    return ",".join(traducidos)

# Home: interfaz
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API: búsqueda
@app.get("/recipes", response_model=List[RecipeSummary])
def search_recipes(
    ingredients: str = Query(...),
    diet: Optional[str] = None,
    intolerances: Optional[str] = None,
    number: int = Query(10, ge=1, le=30)
):
    # traducimos ingredientes y alérgenos al inglés
    ingr_en = traducir_ingredientes(ingredients)
    intol_en = traducir_ingredientes(intolerances) if intolerances else None

    key = f"{ingredients}|{diet}|{intolerances}|{number}"
    if cached := search_cache.get(key):
        return cached

    results: List[RecipeSummary] = []
    # 1) TheMealDB
    r1 = requests.get(
        f"{MEALDB_URL}/filter.php",
        params={"i": ingr_en},
        timeout=5
    ).json()
    if r1.get("meals"):
        for m in r1["meals"]:
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
        if diet: params["diet"] = diet
        if intol_en: params["intolerances"] = intol_en
        sp = requests.get(
            f"{SPOONACULAR_URL}/complexSearch",
            params=params,
            timeout=5
        ).json()
        for r in sp.get("results", []):
            results.append(RecipeSummary(
                id=f"spoon_{r['id']}",
                title=r["title"],
                image=r["image"],
                source="spoonacular"
            ))

    search_cache.set(key, results)
    return results

# API: detalle
@app.get("/recipes/{recipe_id}", response_model=RecipeDetail)
def get_recipe(recipe_id: str, intolerances: Optional[str] = None):
    key = recipe_id + "|" + (intolerances or "")
    if cached := detail_cache.get(key):
        return cached

    # TheMealDB
    if recipe_id.startswith("mealdb_"):
        mid = recipe_id.split("_",1)[1]
        r = requests.get(
            f"{MEALDB_URL}/lookup.php",
            params={"i": mid},
            timeout=5
        ).json()
        meal = r.get("meals", [{}])[0]
        if not meal:
            raise HTTPException(404, "Not found")
        ing, alg = [], []
        for i in range(1,21):
            name = meal.get(f"strIngredient{i}")
            meas = meal.get(f"strMeasure{i}")
            if name:
                txt = f"{meas.strip()} {name.strip()}"
                ing.append(txt)
                ln = name.lower()
                if any(a in ln for a in ["milk","egg","peanut","nut","wheat","soy"]):
                    alg.append(name)
        instr = [s.strip() for s in meal.get("strInstructions","").split(".") if s.strip()]
        detail = RecipeDetail(
            id=recipe_id,
            title=meal["strMeal"],
            image=meal["strMealThumb"],
            source="mealdb",
            ingredients=ing,
            instructions=instr,
            time=None,
            servings=None,
            allergens=alg,
            nutrients=None
        )

    # Spoonacular
    elif recipe_id.startswith("spoon_"):
        sid = recipe_id.split("_",1)[1]
        data = requests.get(
            f"{SPOONACULAR_URL}/{sid}/information",
            params={"apiKey": SPOONACULAR_API_KEY, "includeNutrition": True},
            timeout=5
        ).json()
        if not data.get("id"):
            raise HTTPException(404, "Not found")
        ing = [f"{i['amount']} {i['unit']} {i['name']}" for i in data.get("extendedIngredients",[])]
        instr, nutr = [], {}
        for blk in data.get("analyzedInstructions",[]):
            for s in blk.get("steps",[]):
                instr.append(s["step"])
        for n in data.get("nutrition",{}).get("nutrients",[]):
            if n["name"].lower() in ("calories","fat","carbohydrates","protein"):
                nutr[n["name"]] = f"{n['amount']}{n['unit']}"
        detail = RecipeDetail(
            id=recipe_id,
            title=data["title"],
            image=data["image"],
            source="spoonacular",
            ingredients=ing,
            instructions=instr,
            time=data.get("readyInMinutes"),
            servings=data.get("servings"),
            allergens=(intolerances.split(",") if intolerances else []),
            nutrients=nutr
        )
    else:
        raise HTTPException(400, "Invalid ID")

    detail_cache.set(key, detail)
    return detail

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0",
                port=int(os.getenv("PORT", 8000)), reload=True)
