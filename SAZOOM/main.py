# main.py

import os
import time
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import requests
from dotenv import load_dotenv

# Carga SPOONACULAR_API_KEY desde .env
load_dotenv()
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY", "")

# URLs de las APIs
MEALDB_URL = "https://www.themealdb.com/api/json/v1/1"
SPOONACULAR_URL = "https://api.spoonacular.com/recipes"

# App FastAPI renombrada a Sazoom
app = FastAPI(
    title="Sazoom",
    description="Sazoom: ¿Qué cocino hoy? Búsqueda de recetas por ingredientes, filtros de dieta y alérgenos.",
    version="1.0.0"
)

# Modelos Pydantic
class RecipeSummary(BaseModel):
    id: str
    title: str
    image: str
    source: str  # 'mealdb' o 'spoonacular'

class RecipeDetail(RecipeSummary):
    ingredients: List[str]
    instructions: List[str]
    time: Optional[int]
    servings: Optional[int]
    allergens: Optional[List[str]]
    nutrients: Optional[Dict[str, str]]  # ej. {"Calories":"200 kcal", ...}

# Caché simple con TTL
class SimpleCache:
    def __init__(self, ttl_seconds: int):
        self.ttl = ttl_seconds
        self.store: Dict[str, tuple[float, any]] = {}

    def get(self, key: str):
        entry = self.store.get(key)
        if not entry:
            return None
        timestamp, value = entry
        if time.time() - timestamp > self.ttl:
            del self.store[key]
            return None
        return value

    def set(self, key: str, value: any):
        self.store[key] = (time.time(), value)

# Instancias de caché
search_cache = SimpleCache(ttl_seconds=3600)    # 1 hora
detail_cache = SimpleCache(ttl_seconds=86400)   # 24 horas

# Endpoint de búsqueda de recetas
@app.get("/recipes", response_model=List[RecipeSummary])
def search_recipes(
    ingredients: str = Query(..., description="Ingredientes separados por comas"),
    diet: Optional[str] = Query(None, description="Dieta (vegan, keto, etc.)"),
    intolerances: Optional[str] = Query(None, description="Alérgenos separados por comas"),
    number: int = Query(10, ge=1, le=30, description="Nº máximo de resultados")
):
    cache_key = f"search:{ingredients}:{diet}:{intolerances}:{number}"
    if cached := search_cache.get(cache_key):
        return cached

    results: List[RecipeSummary] = []

    # 1) Intento con TheMealDB
    resp = requests.get(f"{MEALDB_URL}/filter.php", params={"i": ingredients}, timeout=5)
    data = resp.json()
    if resp.ok and data.get("meals"):
        for m in data["meals"]:
            results.append(RecipeSummary(
                id=f"mealdb_{m['idMeal']}",
                title=m["strMeal"],
                image=m["strMealThumb"],
                source="mealdb"
            ))
    else:
        # 2) Fallback a Spoonacular
        params = {
            "apiKey": SPOONACULAR_API_KEY,
            "includeIngredients": ingredients,
            "number": number,
            "addRecipeInformation": True
        }
        if diet:
            params["diet"] = diet
        if intolerances:
            params["intolerances"] = intolerances
        sp = requests.get(f"{SPOONACULAR_URL}/complexSearch", params=params, timeout=5).json()
        for r in sp.get("results", []):
            results.append(RecipeSummary(
                id=f"spoon_{r['id']}",
                title=r["title"],
                image=r["image"],
                source="spoonacular"
            ))

    search_cache.set(cache_key, results)
    return results

# Endpoint de detalle de receta
@app.get("/recipes/{recipe_id}", response_model=RecipeDetail)
def get_recipe(
    recipe_id: str,
    intolerances: Optional[str] = Query(None, description="Alérgenos a destacar")
):
    cache_key = recipe_id + f":{intolerances}"
    if cached := detail_cache.get(cache_key):
        return cached

    # 1) Detalle desde TheMealDB
    if recipe_id.startswith("mealdb_"):
        mid = recipe_id.split("_", 1)[1]
        resp = requests.get(f"{MEALDB_URL}/lookup.php", params={"i": mid}, timeout=5).json()
        meals = resp.get("meals")
        if not meals:
            raise HTTPException(404, "Receta no encontrada")
        meal = meals[0]

        ingredients, allergens = [], []
        for i in range(1, 21):
            ing = meal.get(f"strIngredient{i}")
            meas = meal.get(f"strMeasure{i}")
            if ing and ing.strip():
                txt = f"{meas.strip()} {ing.strip()}"
                ingredients.append(txt)
                low = ing.lower()
                if any(a in low for a in ["milk","egg","peanut","nut","wheat","soy"]):
                    allergens.append(ing.strip())

        instructions = [step.strip() for step in meal["strInstructions"].split(".") if step.strip()]

        detail = RecipeDetail(
            id=recipe_id,
            title=meal["strMeal"],
            image=meal["strMealThumb"],
            source="mealdb",
            ingredients=ingredients,
            instructions=instructions,
            time=None,
            servings=None,
            allergens=allergens or (intolerances.split(",") if intolerances else []),
            nutrients=None
        )

    # 2) Detalle desde Spoonacular
    elif recipe_id.startswith("spoon_"):
        sid = recipe_id.split("_", 1)[1]
        params = {"apiKey": SPOONACULAR_API_KEY, "includeNutrition": True}
        data = requests.get(f"{SPOONACULAR_URL}/{sid}/information", params=params, timeout=5).json()
        if not data.get("id"):
            raise HTTPException(404, "Receta no encontrada")

        ingredients = [
            f"{ing['amount']} {ing['unit']} {ing['name']}"
            for ing in data.get("extendedIngredients", [])
        ]
        instructions = []
        for bloc in data.get("analyzedInstructions", []):
            for step in bloc.get("steps", []):
                instructions.append(step["step"])

        nutrients = {}
        for nut in data.get("nutrition", {}).get("nutrients", []):
            name = nut["name"]
            val = f"{nut['amount']}{nut['unit']}"
            if name.lower() in ("calories","fat","carbohydrates","protein"):
                nutrients[name] = val

        detail = RecipeDetail(
            id=recipe_id,
            title=data["title"],
            image=data["image"],
            source="spoonacular",
            ingredients=ingredients,
            instructions=instructions,
            time=data.get("readyInMinutes"),
            servings=data.get("servings"),
            allergens=intolerances.split(",") if intolerances else [],
            nutrients=nutrients
        )
    else:
        raise HTTPException(400, "ID de receta inválido")

    detail_cache.set(cache_key, detail)
    return detail

# Para ejecutar con `python main.py`
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
