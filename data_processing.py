import config as config
import pandas as pd

def load_data():
    train_raw = pd.read_csv(config.TRAIN_DATA)
    test_raw = pd.read_csv(config.TEST_DATA)
    return train_raw, test_raw

def preprocess_data(df):
    df['ds']= pd.to_datetime(df['date'])
    df['unique_id']= df['store'].astype(str)+'_'+df['item'].astype(str)
    df['y']= df['sales']
    df.drop(columns=['date','store','item','sales'],axis=1,inplace=True)

def get_train_test_split(df):
    train = df.loc[df['ds'] < '2017-01-01']
    test = df.loc[df['ds'] >= '2017-01-01']
    return train, test
