import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import joblib
import json
from azureml.core import Workspace
from azureml.core.model import Model
from sklearn.preprocessing import MinMaxScaler

COLUMN_OF_INTEREST = 'Load'
TRAINING_SAMPLES = 2184
HORIZON = 5
ws = Workspace.from_config()

def init():
    global model
    model_path = Model.get_model_path(model_name = 'SARIMAX-Energy-Forecasting', version = None, _workspace = ws)
    model = joblib.load(model_path)

def run(input_json):
    try:
        energy_load_df = pd.DataFrame(json.loads(input_json))
        energy_load_df = energy_load_df.iloc[0:TRAINING_SAMPLES]
        
        scaler = MinMaxScaler()
        energy_load_df[COLUMN_OF_INTEREST] = scaler.fit_transform(energy_load_df[[COLUMN_OF_INTEREST]])
        
        predictions = model.forecast(steps = HORIZON)
        output_json = pd.Series.to_json(pd.DataFrame(predictions), date_format = 'iso')
        
        return output_json
    
    except Exception as error:
        return str(error)

