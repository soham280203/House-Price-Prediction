import json
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names, but LinearRegression was fitted with feature names")

__locations = None
__data_columns = None
__model = None
__artifacts_loaded = False  # Flag to track if artifacts are loaded

def get_estimated_price(location, sqft, bhk, bath):
    load_saved_artifacts()
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1    
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bhk 
    x[2] = bath
    if loc_index >= 0:
        x[loc_index] = 1
    return round(__model.predict([x])[0], 2)

def get_location_names():
    load_saved_artifacts()
    return __locations

def load_saved_artifacts():
    global __data_columns
    global __locations
    global __model
    global __artifacts_loaded

    if  not __artifacts_loaded:
        print("Loading saved artifacts")
        with open("./Application/Artifacts/Bengaluru_House_Data.pickle", 'rb') as f:
            __model = pickle.load(f)
        with open("./Application/Artifacts/columns.json", 'r') as f:
            __data_columns = json.load(f)['data_columns']
            __locations = __data_columns[3:]
        __artifacts_loaded = True
        print('Loading Artifacts done')

if __name__ == "__main__":
    load_saved_artifacts()
    