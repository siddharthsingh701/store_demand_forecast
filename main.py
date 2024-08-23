import config as config
import utils as utils
import data_processing as dp
import seaborn as sns
import models as models
import pickle
import mlflow
import matplotlib.pyplot as plt
mlflow.set_tracking_uri("http://127.0.0.1:5000")

UNIQUE_ID = '1_13'
if __name__ == "__main__":
    train_df , test_df = dp.load_data()
    dp.preprocess_data(train_df)
    train, test = dp.get_train_test_split(train_df)
    train = train[train['unique_id']==UNIQUE_ID]
    test = test[test['unique_id']==UNIQUE_ID]

    # All Models
    Desc = "Univariate No Major Change"
    models.run_experiment(Desc,'LightGBM',train,test)
    models.run_experiment(Desc,'NBEATS',train,test,input_size=730,h=365)
    models.run_experiment(Desc,'NHITS',train,test,input_size=730,h=365)
    models.run_experiment(Desc,'AutoETS',train,test)
    models.run_experiment(Desc,'Prophet',train,test)

    # HyperParameter Tuning
    Desc= "HyperParameter Tuning"
    models.run_experiment(Desc,'Prophet',train,test,yearly_seasonality=14,seasonality_mode='multiplicative',weekly_seasonality=19)