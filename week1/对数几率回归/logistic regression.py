from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

class Logistic_Regressor(object):
    def __init__(self,solver='newton',c=1.):
        self.beta=0
        self.y_decode=None
        self.y_encode=None
        self.c=c
        self.solver=solver
    
    def fit_newton(self,x,y):
        # preprocess 
        data_len,x_dim=x.shape
        self.beta=np.zeros(x_dim+1)
        x=np.column_stack((x,np.ones(data_len)))

        # Newton's method
        while True:
            # calculate Sigmoid
            p_exp=np.exp(x.dot(self.beta))
            p_exp=np.clip(p_exp,0,1e14) 
            p1=p_exp/(1+p_exp)

            # calculate first and second order derivative
            l1=-self.c*np.sum(x.T*(y-p1),axis=1)
            l1[:-1]+=self.beta[:-1]
            
            # Stop condition
            if np.max(np.abs(l1))<1e-1:
                break

            l2=self.c*np.dot(p1*(1-p1)*x.T,x)
            for i in range(l2.shape[0]):
                l2[i,i]+=1
            l2=np.linalg.pinv(l2)
            
            # step forward
            step=-l2.dot(l1)
            # print(self.beta.shape,step.shape)
            self.beta+=step

    def fit(self,x,y):
        # preprocess 
        x=np.array(x,dtype=float)
        y=np.array(y)
        self.y_decode=np.unique(y)
        self.y_encode=dict([[b,a] for a,b in enumerate(self.y_decode)])
        y=np.array([self.y_encode[a] for a in y])

        if self.solver=='newton':
            self.fit_newton(x,y)
        elif self.solver=='dfp':
            self.fit_DFP(x,y)
        elif self.solver=='bfgs':
            self.fit_BFGS(x,y)
          

    def predict(self,x):
        x=np.array(x,dtype=float)
        data_len,_=x.shape
        x=np.column_stack((x,np.ones(data_len)))
        p_exp=np.exp(x.dot(self.beta))
        p_exp=np.clip(p_exp,0,1e14) 
        p1=p_exp/(1+p_exp)
        res=np.around(p1).astype(int)
        return [self.y_decode[a] for a in res]
    
    def fit_DFP(self,x,y): #DFP拟牛顿法
        data_len,x_dim=x.shape
        self.beta=np.zeros(x_dim+1)
        x=np.column_stack((x,np.ones(data_len)))

        n = len(x[0])
        theta=np.zeros((n,1))
        y=np.mat(y).T
        Gk=np.eye(n,n)
        grad_last = self.c*np.dot(x.T,self.sigmoid(np.dot(x,theta))-y)+theta
        cost=[]
        # for it in range(100):
        while True:
            pk = -1 * Gk.dot(grad_last)
            rate=self.alphA(x,y,theta,pk)
            theta = theta + rate * pk
            grad= self.c*np.dot(x.T,self.sigmoid(np.dot(x,theta))-y)+theta
            delta_k = rate * pk
            y_k = (grad - grad_last)
            Pk = delta_k.dot(delta_k.T) / (delta_k.T.dot(y_k))
            Qk= Gk.dot(y_k).dot(y_k.T).dot(Gk) / (y_k.T.dot(Gk).dot(y_k)) * (-1)
            Gk += Pk + Qk
            grad_last = grad
            if np.max(np.abs(grad_last))<1e-1:
                break
            cost.append(np.sum(grad_last))
        self.beta=np.array(theta.T)[0]

    def fit_BFGS(self,x,y): #DFP拟牛顿法
        data_len,x_dim=x.shape
        self.beta=np.zeros(x_dim+1)
        x=np.column_stack((x,np.ones(data_len)))

        n = len(x[0])
        theta=np.zeros((n,1))
        y=np.mat(y).T
        Gk=np.eye(n,n)
        grad_last = self.c*np.dot(x.T,self.sigmoid(np.dot(x,theta))-y)+theta
        cost=[]
        # for it in range(100):
        while True:
            pk = -1 * Gk.dot(grad_last)
            rate=self.alphA(x,y,theta,pk)
            theta = theta + rate * pk
            grad= self.c*np.dot(x.T,self.sigmoid(np.dot(x,theta))-y)+theta
            delta_k = rate * pk
            y_k = (grad - grad_last)

            rou= delta_k.T.dot(y_k)

            Pk = delta_k.dot(delta_k.T) / rou
            Qk= (np.identity(n)-delta_k.dot(y_k.T) / rou).dot(Gk).dot((np.identity(n)-y_k.dot(delta_k.T) / rou))
            Qk= Gk.dot(y_k).dot(y_k.T).dot(Gk) / (y_k.T.dot(Gk).dot(y_k)) * (-1)
            Gk += Pk + Qk
            grad_last = grad
            if np.max(np.abs(grad_last))<1e-1:
                break
            cost.append(np.sum(grad_last))
        self.beta=np.array(theta.T)[0]

    def sigmoid(self,x): #simoid 函数
        return 1.0/(1+np.exp(-x))
    
    def alphA(self,x,y,theta,pk): #选取前20次迭代cost最小的alpha
        # return 1
        c=float("inf")
        t=theta
        for k in range(1,200):
            a=k**2/50
            theta = t + a * pk
            f= np.sum(self.c*np.dot(x.T,self.sigmoid(np.dot(x,theta))-y)+theta)
            if abs(f)>c:
                break
            c=abs(f)
            alpha=a
        return alpha

def load_data1():
    data=pd.read_csv('./data/wdbc.data',index_col=0,header=None)
    return data.iloc[:,1:],data.iloc[:,0]

def load_data2():
    data2=pd.read_csv('./data/pd_speech_features.csv',index_col=0)
    data2.columns=data2.iloc[0,:]
    data2=data2[1:]
    return data2.iloc[:,:-1],data2.iloc[:,-1]

def sklearn_logistic_regression(x,y,self_done=False,solver='newton',print=True):
    minmax=MinMaxScaler()
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=20,test_size=0.2)
    if self_done:
        model=Logistic_Regressor(solver=solver)
    else:
        model=LogisticRegression()
    x_train=minmax.fit_transform(x_train)
    model.fit(x_train,y_train)
    x_test=minmax.transform(x_test)
    pred=model.predict(x_test)
    if print:
        print(classification_report(y_test,pred))


def adj_lamda(x,y):
    minmax=MinMaxScaler()
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=20,test_size=0.2)
    x_train=minmax.fit_transform(x_train)
    x_test=minmax.transform(x_test)
    c=[]
    acc=[]
    precision=[]
    recall=[]
    f1=[]
    acc2=[]
    precision2=[]
    recall2=[]
    f12=[]
    for i in np.arange(11,-5,-0.2):
        C=10**i
        c.append(1/C)
        model=LogisticRegression(C=C,max_iter=500)
        model.fit(x_train,y_train)
        pred=model.predict(x_test)
        res=classification_report(y_test,pred,output_dict=True)
        acc.append(res['accuracy'])
        precision.append(res['weighted avg']['precision'])
        recall.append(res['weighted avg']['recall'])
        f1.append(res['weighted avg']['f1-score'])

        pred=model.predict(x_train)
        res=classification_report(y_train,pred,output_dict=True)
        acc2.append(res['accuracy'])
        precision2.append(res['weighted avg']['precision'])
        recall2.append(res['weighted avg']['recall'])
        f12.append(res['weighted avg']['f1-score'])

    plt.subplot(121)
    plt.title('Testing set')
    plt.xlabel('λ')
    plt.ylabel('Score')
    plt.xscale('log')
    plt.plot(c,acc,label='acc')
    plt.plot(c,precision,label='precision')
    plt.plot(c,recall,label='recall')
    plt.plot(c,f1,label='f1-score')
    plt.legend()
    plt.subplot(122)
    plt.title('Training set')
    plt.xlabel('λ')
    plt.ylabel('Score')
    plt.xscale('log')
    plt.plot(c,acc2,label='acc')
    plt.plot(c,precision2,label='precision')
    plt.plot(c,recall2,label='recall')
    plt.plot(c,f12,label='f1-score')
    plt.legend()
    plt.show()

def timeit(x,y,self_done=False,solver='newton'):
    t=time.time()
    for i in range(10):
        sklearn_logistic_regression(x,y,self_done=self_done,solver=solver,print=False)
    t=time.time()-t
    t/=10
    return t



# x,y=load_data1()
# sklearn_logistic_regression(x,y,False)
# x,y=load_data2()
# sklearn_logistic_regression(x,y,False)

# x,y=load_data1()
# sklearn_logistic_regression(x,y,True)
# x,y=load_data2()
# sklearn_logistic_regression(x,y,True)

# x,y=load_data2()
# adj_lamda(x,y)

x,y=load_data2()
res=[]
res.append(timeit(x,y))
res.append(timeit(x,y,self_done=True,solver='newton'))
res.append(timeit(x,y,self_done=True,solver='dfp'))
res.append(timeit(x,y,self_done=True,solver='bfgs'))
print(res)

# [0.48047819137573244, 3.2909286499023436, 6.13991277217865, 15.228477263450623]
# [0.021604013442993165, 0.016522145271301268, 0.1388854503631592, 0.13941380977630616]

# [0.023927068710327147, 0.01493988037109375, 0.018261981010437012, 0.01968522071838379]

# [0.410368537902832, 3.058761167526245, 1.8805846452713013, 4.254945707321167]