import json

with open("cities.json", "r") as f:
    cities = json.load(f)

def get_city_by_code(code: str) -> str:
    """Get city by code"""
    if code in cities:
        return cities[code]
    else:
        return code