import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
# 返回预处理后的pandas数据

def load_iris():
    iris_columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
    iris_data = pd.read_csv('./data/iris/iris.data', sep=',', names=iris_columns)
    return iris_data

def load_forestfires():
    columns = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']
    forestfires_data = pd.read_csv('./data/forestfires/forestfires.csv', sep=',', names=columns,skiprows=1)
    forestfires_data.drop(['month', 'day'], axis=1, inplace=True)
    return forestfires_data

def load_adult():
    # 加载数据，每行为一个样本，属性以逗号分开
    train_data_file = './data/adult/adult.data'
    test_data_file = './data/adult/adult.test'
    columns = ['age', 'workclass', 'fnlgwt', 'education', 'education-num', 'marital-status',
               'occupation', 'relationship', 'race', 'sex', 'capital-gain',
               'capital-loss', 'hours-per-week', 'native-country', 'income']
    train_df = pd.read_csv(train_data_file, sep=', ', names=columns, engine='python')
    test_df = pd.read_csv(test_data_file, sep=', ', names=columns, skiprows=1, engine='python')  # 忽略第一行
    # ================= 数据预处理 =================
    # ================= 数据可视化 =================
    # print(train_df.info())
    # print(test_df.info())
    # ================= 扔掉无用属性（列） =================
    train_df.drop(['fnlgwt', 'education'], axis=1, inplace=True)
    test_df.drop(['fnlgwt', 'education'], axis=1, inplace=True)
    # ================= 处理缺失值 =================
    # 这个数据集中，缺失值用'?'来表示，在打印了train_df的信息之后，发现6列数值类型的属性在dataframe中也是数值类型，所以这几列没有缺失值。
    # 如果有缺失值，可能要将这一列转换为数值类型。
    # 简单处理：扔掉缺失的行
    train_df.replace('?', np.nan, inplace=True)
    test_df.replace('<=50K.', '<=50K', inplace=True)
    test_df.replace('>50K.', '>50K', inplace=True)
    # print(train_df.iloc[14].values)
    train_df.dropna(axis=0, how='any', inplace=True)
    test_df.replace('?', np.nan, inplace=True)
    test_df.dropna(axis=0, how='any', inplace=True)
    # print(test_df.info())

    # 处理非数值数据
    # X = train_df.select_dtypes(include=[object])
    # s = 0
    # for i in X.columns:
    #     s = s + train_df[i].value_counts().size
    #     print(i, train_df[i].value_counts().size)
    # print('s=', s)
    train_X = train_df.drop('income', axis=1)
    train_Y = train_df[['income']].values.reshape(-1, )
    attributes = train_X.select_dtypes(include=[object]).columns.tolist()
    x_enc = ColumnTransformer([('attributes', OneHotEncoder(), attributes)], remainder='passthrough').fit(train_X)
    # print(x_enc.transformers_[0][1].categories_)
    y_enc = LabelEncoder().fit(train_Y)
    

    train_X = x_enc.transform(train_X).toarray()
    train_Y = y_enc.transform(train_Y)
    test_X = x_enc.transform(test_df.drop('income', axis=1)).toarray()
    test_Y = y_enc.transform(test_df[['income']].values.reshape(-1, ))
    # print(test_X) 
    return train_X, test_X, train_Y, test_Y

load_adult()