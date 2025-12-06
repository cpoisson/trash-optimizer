from bs4 import BeautifulSoup
import pandas as pd

# Charger le fichier HTML rendu côté client
with open("Ecosystem-ampoules-nantes.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

data = []
# Chaque point de collecte est un <article class="cp-card">
for card in soup.select("article.cp-card"):
    name_el = card.select_one("h3.cp-card-title")
    addr_el = card.select_one("address.cp-card-address")
    distance_el = card.select_one("p.cp-card-distance")
    types_el = card.select("ul.cp-card-equipments li.tag")  # liste des types
    link_el = card.select_one("a.cp-card-link")

    name = name_el.get_text(strip=True) if name_el else ""
    address = addr_el.get_text(separator=" ", strip=True) if addr_el else ""
    distance = distance_el.get_text(strip=True) if distance_el else ""
    types = [t.get_text(strip=True) for t in types_el] if types_el else []
    link = link_el['href'] if link_el else ""

    data.append({
        "name": name,
        "address": address,
        "distance": distance,
        "types": ", ".join(types),
        "link": link
    })

# Créer le DataFrame et sauvegarder en CSV
df = pd.DataFrame(data)
print(df)
df.to_csv("ecosystem-ampoules-nantes.csv", index=False)
