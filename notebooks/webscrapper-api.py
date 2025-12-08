import requests
import pandas as pd

url = "https://www.ecosystem.eco/recherche/markers"

params = {
    "product": "Piles",
    "location": "Nantes, France",
    "lat": 47.218371,
    "lon": -1.553621,
    "city": "Nantes",
    "country": "France",
    "department": "Loire-Atlantique"
}

# Appel API
r = requests.get(url, params=params)
data_json = r.json() # 'markers' contient la liste des points
markers = data_json.get("markers", [])
rows = []
for m in markers:
    coords = m.get("coords", {})
    lat = coords.get("lat")
    lon = coords.get("lng")

    rows.append({
        "name": m.get("name", ""),
        "address": m.get("address", ""),
        "lat": lat,
        "lon": lon,
        "distance": m.get("distance", ""),
        "types": ", ".join(m.get("types", [])),
        "id": m.get("id", ""),
    })

df = pd.DataFrame(rows)
df.to_csv(f"{params['city']}_{params['product']}_api_scrapped.csv", index=False)
