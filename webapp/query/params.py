# category_mapping = {
#     "food_organics": "Is_Food_enabled = 1",
#     "cardboard": "Is_Cardboard_enabled = 1",
#     "glass": "Is_Glass_enabled = 1",
#     "metal":"Is_Metal_enabled = 1",
#     "paper": "Is_Paper_enabled = 1",
#     "plastic": "Is_Plastic_enabled = 1",
#     "textile_trash": "Is_Textile_enabled=1",
#     "vegetation":"Is_Vegetation_enabled =1",
#     "miscellaneous_trash":"Is_Miscellanous_Trash_enabled =1",
#     "mirror":"Is_Glass_enabled = 1",# TO be changed
#     "car_battery":"Is_Miscellanous_Trash_enabled =1",# TO be changed
#     "neon":"Is_Neon_enabled=1",
#     "pharmacy":"Is_Pharmacy_enabled=1",
#     "tire":"Is_Tire_enabled=1",
#     "printer_cartridge":"Is_Cartridge_enabled=1",
#     "wood":"Is_Miscellanous_Trash_enabled =1", # TO be changed
#     "ressourcerie":"Is_Ressourcerie_enabled=1",
#     "light_bulb":"Is_Lamp_Light_enabled=1",
#     "battery":"Is_Lamp_Light_enabled=1"     # TO be changed
# }

category_mapping = {
    "food_organics":        "accepts_food = 1",
    "cardboard":            "accepts_cardboard = 1",
    "glass":                "accepts_glass = 1",
    "metal":                "accepts_metal = 1",
    "paper":                "accepts_paper = 1",
    "plastic":              "accepts_plastic = 1",
    "textile_trash":        "accepts_textile = 1",
    "vegetation":           "accepts_vegetation = 1",
    "miscellaneous_trash":  "accepts_miscellaneous = 1",

    # Ancien "mirror" → probablement verre
    "mirror":               "accepts_glass = 1",

    # Ancien "car_battery" → colonne explicite
    "car_battery":          "accepts_car_battery = 1",

    # Correspondances directes
    "neon":                 "accepts_neon = 1",
    "pharmacy":             "accepts_pharmacy = 1",
    "tire":                 "accepts_tire = 1",
    "printer_cartridge":    "accepts_cartridge = 1",
    "wood":                 "accepts_wood = 1",
    "ressourcerie":         "accepts_ressourcerie = 1",

    # Lamp / ampoules
    "light_bulb":           "accepts_lamp = 1",

    # "battery" (piles) → colonne dédiée
    "battery":              "accepts_pile = 1"
}
