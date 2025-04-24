import os, time, requests
from typing import List, Optional, Dict
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY", "")

MEALDB_URL = "https://www.themealdb.com/api/json/v1/1"
SPOONACULAR_URL = "https://api.spoonacular.com/recipes"
LIBRETRANSLATE_URL = "https://libretranslate.de/translate"
MYMEMORY_URL = "https://api.mymemory.translated.net/get"

app = FastAPI(title="Sazoom")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def _lt(text: str, src: str, tgt: str) -> str:
    try:
        r = requests.post(
            LIBRETRANSLATE_URL,
            json={"q": text, "source": src, "target": tgt, "format": "text"},
            timeout=3
        ).json()
        return r.get("translatedText", text)
    except:
        try:
            p = {"q": text, "langpair": f"{src}|{tgt}"}
            r = requests.get(MYMEMORY_URL, params=p, timeout=3).json()
            return r.get("responseData", {}).get("translatedText", text)
        except:
            return text

def translate_text(text: str) -> str:
    return _lt(text, "en", "es")

def translate_list(items: List[str]) -> List[str]:
    return [translate_text(i) for i in items] if items else []

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
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/recipes", response_model=List[RecipeSummary])
def search_recipes(
    ingredients: str = Query(...),
    diet: Optional[str] = None,
    intolerances: Optional[str] = None,
    number: int = Query(10, ge=1, le=30)
):
    key = f"{ingredients}|{diet}|{intolerances}|{number}"
    if cached := search_cache.get(key):
        return cached

    results: List[RecipeSummary] = []
    # 1) TheMealDB
    r1 = requests.get(f"{MEALDB_URL}/filter.php",
                      params={"i": ingredients}, timeout=5).json()
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
            "includeIngredients": ingredients,
            "number": number,
            "addRecipeInformation": True
        }
        if diet: params["diet"] = diet
        if intolerances: params["intolerances"] = intolerances
        sp = requests.get(f"{SPOONACULAR_URL}/complexSearch",
                          params=params, timeout=5).json()
        for r in sp.get("results", []):
            results.append(RecipeSummary(
                id=f"spoon_{r['id']}",
                title=r["title"],
                image=r["image"],
                source="spoonacular"
            ))

    # traducir sólo los títulos EN→ES
    for r in results:
        r.title = translate_text(r.title)

    search_cache.set(key, results)
    return results

@app.get("/recipe/{recipe_id}")
def recipe_page(request: Request, recipe_id: str, intolerances: Optional[str] = None):
    detail = _fetch_detail(recipe_id, intolerances)
    return templates.TemplateResponse("detail.html", {
        "request": request,
        "recipe": detail
    })

@app.get("/recipes/{recipe_id}", response_model=RecipeDetail)
def get_recipe(recipe_id: str, intolerances: Optional[str] = None):
    return _fetch_detail(recipe_id, intolerances)

def _fetch_detail(recipe_id: str, intolerances: Optional[str]):
    key = f"{recipe_id}|{intolerances or ''}"
    if cached := detail_cache.get(key):
        return cached

    # TheMealDB
    if recipe_id.startswith("mealdb_"):
        mid = recipe_id.split("_",1)[1]
        data = requests.get(f"{MEALDB_URL}/lookup.php",
                            params={"i": mid}, timeout=5).json()\
                       .get("meals", [{}])[0]
        if not data:
            raise HTTPException(404, "Not found")
        ing, alg = [], []
        for i in range(1,21):
            n = data.get(f"strIngredient{i}")
            m = data.get(f"strMeasure{i}")
            if n:
                ing.append(f"{m.strip()} {n.strip()}")
                if any(a in n.lower() for a in
                       ["milk","egg","peanut","nut","wheat","soy"]):
                    alg.append(n)
        instr = [s.strip() for s in
                 data.get("strInstructions","").split(".")
                 if s.strip()]
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
            params={"apiKey": SPOONACULAR_API_KEY,
                    "includeNutrition": True},
            timeout=5
        ).json()
        if not d.get("id"):
            raise HTTPException(404, "Not found")
        ing = [f"{i['amount']} {i['unit']} {i['name']}"
               for i in d.get("extendedIngredients", [])]
        instr = []
        for b in d.get("analyzedInstructions", []):
            instr += [s["step"] for s in b.get("steps", [])]
        nutr = {
            n["name"]: f"{n['amount']}{n['unit']}"
            for n in d.get("nutrition", {})\
                        .get("nutrients", [])
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
        raise HTTPException(400, "Invalid ID")

    # traducir sólo por ítem EN→ES
    detail.title = translate_text(detail.title)
    detail.ingredients = translate_list(detail.ingredients)
    detail.instructions = translate_list(detail.instructions)
    detail.allergens = translate_list(detail.allergens or [])
    if detail.nutrients:
        keys = list(detail.nutrients.keys())
        keys_es = translate_list(keys)
        detail.nutrients = {ke: detail.nutrients[k]
                            for ke,k in zip(keys_es, keys)}

    detail_cache.set(key, detail)
    return detail

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0",
                port=int(os.getenv("PORT", 8000)), reload=True)
