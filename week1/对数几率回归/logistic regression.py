from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

class Logistic_Regressor(object):
    def __init__(self,c=1.):
        self.beta=0
        self.y_decode=None
        self.y_encode=None
        self.c=c
        self.max=None
        self.min=None
    
    def fit(self,x,y):
        # preprocess 
        x=np.array(x,dtype=float)
        y=np.array(y)
        data_len,x_dim=x.shape
        self.y_decode=np.unique(y)
        self.y_encode=dict([[b,a] for a,b in enumerate(self.y_decode)])
        y=np.array([self.y_encode[a] for a in y])

        # min-max standardize
        self.max=np.max(x,axis=0)
        self.min=np.min(x,axis=0)
        x=(x-self.min)/(self.max-self.min)
        

        self.beta=np.ones(x_dim+1)/(x_dim+1)
        x=np.column_stack((x,np.ones(data_len)))
        l_p1=-1

        # Newton's method
        while True:
            # calculate Sigmoid
            p_exp=np.exp(x.dot(self.beta))
            p_exp=np.clip(p_exp,0,1e14) 
            p1=p_exp/(1+p_exp)

            # Stop condition
            if np.max(np.abs(p1-l_p1))<5e-5:
                break
            l_p1=p1

            # calculate first and second order derivative
            l1=-self.c*np.sum(x.T*(y-p1),axis=1)
            l2=self.c*np.dot(p1*(1-p1)*x.T,x)
            l1[:-1]+=self.beta[:-1]
            for i in range(l2.shape[0]):
                l2[i,i]+=1

            # step forward
            step=np.linalg.pinv(l2).dot(l1)
            self.beta-=step

    def predict(self,x):
        x=np.array(x,dtype=float)
        data_len,_=x.shape
        x=(x-self.min)/(self.max-self.min)
        x=np.column_stack((x,np.ones(data_len)))
        p_exp=np.exp(x.dot(self.beta))
        p_exp=np.clip(p_exp,0,1e14) 
        p1=p_exp/(1+p_exp)
        res=np.around(p1).astype(int)
        return [self.y_decode[a] for a in res]

def load_data1():
    data=pd.read_csv('./data/wdbc.data',index_col=0,header=None)
    return data.iloc[:,1:],data.iloc[:,0]

def load_data2():
    data2=pd.read_csv('./data/pd_speech_features.csv',index_col=0)
    data2.columns=data2.iloc[0,:]
    data2=data2[1:]
    return data2.iloc[:,:-1],data2.iloc[:,-1]

def sklearn_logistic_regression(x,y,self_done=False):
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=20,test_size=0.2)
    if self_done:
        model=Logistic_Regressor()
    else:
        model=LogisticRegression()
    model.fit(x_train,y_train)
    pred=model.predict(x_test)

    print(classification_report(y_test,pred))


x,y=load_data1()
sklearn_logistic_regression(x,y,False)
x,y=load_data2()
sklearn_logistic_regression(x,y,False)

x,y=load_data1()
sklearn_logistic_regression(x,y,True)
x,y=load_data2()
sklearn_logistic_regression(x,y,True)
