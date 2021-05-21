import numpy as np
import pandas as pd

def data_process():
    # bikes_hour_df_raw = pd.read_csv('../Bike-Sharing-Dataset/hour.csv')
    bikes_day_df_raw = pd.read_csv('../Bike-Sharing-Dataset/day.csv')
    # 获得只有属性yr、workingday、atemp、hum、windspeed、cnt的子表
    bikes_day_df = bikes_day_df_raw.drop(
        ['instant', 'dteday', 'season', 'mnth', 'holiday', 'weekday', 'weathersit', 'temp', 'casual', 'registered'],
        axis=1)
    ##数据类型、数量和空值
    bikes_day_df.info()
    ##某一列数据信息，包含max、min等
    bikes_day_df['cnt'].describe()
    ###转换为dataframe
    # col_n = ['atemp','hum','windspeed']
    # bikes_day_df = pd.DataFrame(bikes_day_df_raw,columns = col_n)
    yr0 = pd.DataFrame(bikes_day_df)
    yr1 = pd.DataFrame(bikes_day_df)
    yr0 = yr0[(yr0['yr'] == 0) & (yr0['workingday'] == 1)]
    yr1 = yr1[(yr1['yr'] == 1) & (yr1['workingday'] == 1)]
    ##转化成array
    col_n = ['atemp', 'hum', 'windspeed', 'cnt']
    yr0 = pd.DataFrame(yr0, columns=col_n).values
    yr1 = pd.DataFrame(yr1, columns=col_n).values
    train_X = yr0[:, :3]
    train_Y = yr0[:, 3]
    test_X = yr1[:, :3]
    test_Y = yr1[:, 3]
    return train_X,train_Y,test_X,test_Y


