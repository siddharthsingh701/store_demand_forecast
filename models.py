
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS,TFT
from sklearn.metrics import r2_score
from statsforecast.models import AutoARIMA,AutoETS
from statsforecast import StatsForecast
import pickle
from lightgbm import LGBMRegressor
from prophet import Prophet

class Forecasting_Model:
    def __init__(self,model_name,train,test,**params):
        self.model_name = model_name
        self.params = params
        self.train = train
        self.test = test
    def fit(self):
        if self.model_name == 'NBEATS':
            models = [NBEATS(**self.params)]
            self.model = NeuralForecast(models=models, freq='D')
            self.model.fit(self.train)
        elif self.model_name == 'NHITS':
            models = [NHITS(**self.params)]
            self.model = NeuralForecast(models=models, freq='D')
            self.model.fit(self.train)
        elif self.model_name == 'TFT':
            models = [TFT(**self.params)]
            self.model = NeuralForecast(models=models, freq='D')
            self.model.fit(self.train)
        elif self.model_name == 'AutoARIMA':
            models = [AutoARIMA(**self.params)]
            self.model = StatsForecast(models=models, freq='D')
            self.model.fit(self.train)
        elif self.model_name == 'LightGBM':
            self.model = LGBMRegressor(**self.params)
            self.model.fit(self.train.index.values.reshape(-1,1),self.train['y'].values)
        elif self.model_name == 'AutoETS':
            models = [AutoETS(**self.params)]
            self.model = StatsForecast(models=models, freq='D')
            self.model.fit(self.train)
        elif self.model_name == 'Prophet':
            self.model = Prophet(**self.params)
            self.model.fit(self.train.drop('unique_id',axis=1))

    def forecast(self,horizon=30):
        if self.model_name in ['NBEATS','NHITS','TFT']:
            self.forecast_ = self.model.predict().reset_index()
            self.forecast_.columns = ['unique_id','ds','forecast']
            self.forecast_ = self.forecast_['forecast'].values
            return self.forecast_
        elif self.model_name in ['AutoARIMA','AutoETS']:
            self.forecast_= self.model.predict(h=horizon).reset_index()
            self.forecast_.columns = ['unique_id','ds','forecast']
            self.forecast_ = self.forecast_['forecast'].values
            return self.forecast_
        elif self.model_name == 'LightGBM':
            self.forecast_ = self.model.predict(self.test.index.values.reshape(-1,1))
            return self.forecast_
        elif self.model_name == 'Prophet':
            self.forecast_= self.model.predict(self.test.drop('unique_id',axis=1))['yhat'].values
            return self.forecast_

    def wmape(self):
        y_true = self.test['y'].values
        y_pred = self.forecast_

        try:
            return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)
        except:
            return np.nan
    
    def plot_forecast(self,show=False):
        plt.figure(figsize=(20,8))
        # plt.plot(self.train.index,self.train['y'],label='Training Data')
        plt.plot(self.test.index,self.test['y'],label='Test Data')
        plt.plot(self.test.index,self.forecast_,label='Forecast')
        if show:
            plt.show()
        else:
            plt.savefig('forecast_image.png')
            plt.close()

def run_experiment(desc,model_name,train_data,test_data,**params):
    mlflow.set_experiment(desc)
    with mlflow.start_run():
        model = Forecasting_Model(model_name=model_name,train=train_data,test=test_data,**params)
        model.fit()
        pred = model.forecast(horizon=365)
        wmape = model.wmape()
        model.plot_forecast()
        with open('model.pkl','wb') as file:
            pickle.dump(model,file)
        mlflow.log_metric("WMAPE", wmape)
        mlflow.log_params(params)
        mlflow.log_param("Model_Name",model_name)
        mlflow.log_param("Description",desc)
        mlflow.log_artifact('forecast_image.png')
        mlflow.log_artifact('model.pkl')