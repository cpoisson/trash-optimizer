import pandas as pd
DUMMY_PREDICT_RESULT = [{
    'bio_hazard':0.8},
    {'metal':0.2}
]

DUMMY_DROP_OFF = {"lat": 47.2138, "lon": -1.5487}

DUMMY_START_POINT  = {"lat": 47.2109096, "lon":-1.5501935}

DUMMY_LIST_TRASH = ['metal','bio']

DUMMY_DATAFRAME_GEOLOC =  pd.DataFrame([
    {"name": "Spot_1", "lat": 47.2129, "lon": -1.5479, "trash_type": "bio_hazard"},
    {"name": "Spot_2", "lat": 47.2142, "lon": -1.5494, "trash_type": "metal"},
    {"name": "Spot_3", "lat": 47.2116, "lon": -1.5481, "trash_type": "metal"},
    {"name": "Spot_4", "lat": 47.2131, "lon": -1.5507, "trash_type": "bio_hazard"},
    {"name": "Spot_5", "lat": 47.2122, "lon": -1.5511, "trash_type": "metal"},
    {"name": "Spot_6", "lat": 47.2107, "lon": -1.5498, "trash_type": "bio_hazard"},
    {"name": "Spot_7", "lat": 47.2115, "lon": -1.5519, "trash_type": "metal"},
    {"name": "Spot_8", "lat": 47.2127, "lon": -1.5502, "trash_type": "bio_hazard"},
])

def dummy_predict():
    return DUMMY_PREDICT_RESULT

def dummy_get_drop_off():
    return DUMMY_DROP_OFF

def dummy_get_loc(list_trash = DUMMY_LIST_TRASH):
    return DUMMY_DATAFRAME_GEOLOC
