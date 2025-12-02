DUMMY_PREDICT_RESULT = [{
    'bio_hazard':0.8},
    {'metal':0.2}
]

DUMMY_DROP_OFF = {"lat": 47.2138, "lon": -1.5487}

DUMMY_START_POINT  = {"lat": 47.2109096, "lon":-1.5501935}
def dummy_predict():
    return DUMMY_PREDICT_RESULT

def dummy_get_drop_off():
    return DUMMY_DROP_OFF
