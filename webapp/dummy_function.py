import pandas as pd

DUMMY_PREDICT_RESULT = [
    {"class": "food_organics",        "confidence": 0.80},
    {"class": "cardboard",            "confidence": 0.70},
    {"class": "glass",                "confidence": 0.90},
    {"class": "metal",                "confidence": 0.60},
    {"class": "paper",                "confidence": 0.80},
    {"class": "plastic",              "confidence": 0.70},
    {"class": "textile_trash",        "confidence": 0.75},
    {"class": "vegetation",           "confidence": 0.85},
    {"class": "miscellaneous_trash",  "confidence": 0.90},
    {"class": "mirror",               "confidence": 0.80},
    {"class": "car_battery",          "confidence": 0.70},
    {"class": "neon",                 "confidence": 0.90},
    {"class": "pharmacy",             "confidence": 0.60},
    {"class": "tire",                 "confidence": 0.65},
    {"class": "printer_cartridge",    "confidence": 0.70},
    {"class": "wood",                 "confidence": 0.80},
    {"class": "ressourcerie",         "confidence": 0.90},

    # Classes présentes dans ton mapping mais absentes de ta liste
    {"class": "light_bulb",           "confidence": 0.85},
    # {"class": "battery",              "confidence": 0.88}
]

DUMMY_DROP_OFF = {"lat": 47.2138, "lon": -1.5487}

DUMMY_START_POINT  = {"lat": 47.2109096,
                      "lon":-1.5501935,
                      "trash_type":"User Start Point",
                      "distance": 0}

DUMMY_LIST_TRASH = ['metal','bio']

DUMMY_PROBABILITY_THRESHOLD = 0.1

DUMMY_DATAFRAME_GEOLOC = pd.DataFrame([
    {"name": "Spot_1", "lat": 47.2180, "lon": -1.5530, "trash_type": "Food Organics"},  # nord-ouest
    {"name": "Spot_2", "lat": 47.2250, "lon": -1.5370, "trash_type": "Metal"},       # nord-est
    {"name": "Spot_3", "lat": 47.2000, "lon": -1.5400, "trash_type": "Food Organics"},       # sud-est
    {"name": "Spot_4", "lat": 47.2050, "lon": -1.5550, "trash_type": "Glass"},  # sud-ouest
    {"name": "Spot_5", "lat": 47.2300, "lon": -1.5500, "trash_type": "Glass"},       # nord
    {"name": "Spot_6", "lat": 47.1950, "lon": -1.5450, "trash_type": "Cardboard"},  # sud
    {"name": "Spot_7", "lat": 47.2100, "lon": -1.5300, "trash_type": "metal"},       # est
    {"name": "Spot_8", "lat": 47.2150, "lon": -1.5650, "trash_type": "bio_hazard"},  # ouest
])

def dummy_predict():
    return DUMMY_PREDICT_RESULT

def dummy_get_drop_off():
    return DUMMY_DROP_OFF

def dummy_get_loc(list_trash = DUMMY_LIST_TRASH):
    return DUMMY_DATAFRAME_GEOLOC
