import numpy as np

##数据去重
def unique(arr):
    n = len(arr[:])
    i = 0
    while(i<n):
        for j in range(0,n):
            if i == j:
                continue
            if arr[i] == arr[j]:
                arr = np.delete(arr,j,0)
                n = n - 1
                i = i - 1
                break
        i = i + 1
    return arr

##构造二维pareto前沿，x为x轴上界，y为y轴上界
def Append(arr,x,y):
    n = len(arr[:])
    for k in range(0,n):
        if k==n-1:
            if k==0:
                arr = np.concatenate((arr,[[arr[0][0],y]]))#需调整边界Y
                arr = np.concatenate((arr,[[x,arr[k][1]]]))#需调整边界X
                #print(arr,n-1)
                #input()
                break
            else:
                arr = np.concatenate((arr,[[x,arr[k][1]]]))#需调整边界X
                #print(arr,n-1)
                #input()
                break
        if arr[k][0]==arr[k+1][0]:
            continue
        if k==0:
            arr = np.concatenate((arr,[[arr[0][0],y]]))#需调整边界Y
            #print(arr,0)
            #input()
        arr = np.concatenate((arr,[[arr[k+1][0],arr[k][1]]]))
        #print(arr,3)
        #input()
    return arr


def pareto(arr):
    dim = arr.shape[1]#计算目标值维度
    if dim <= 1:
        print("维数小于等于2，无pareto")
        return
    bubble_sort(arr)
    n = len(arr[:])
    i = 0
    while(i<n):
        for j in range(i,n):
                if np.sum((arr[j]-arr[i]) > 0) <= 0:
                    arr = np.delete(arr,i,0)
                    n = n - 1
                    i = i - 1
                    break
        i = i + 1
    return arr


##从第i维开始排序,稳定排序
def bubble_sort(arr,i = 0):
    dim = arr.shape[1]#计算目标值维度
    if i > dim - 1:
        print("排序维度超出数据维度")
        return
    t = np.zero(dim)
    n = len(arr[:])
    while(i <= dim-1):
        for j in range(0,n-1):
            for k in range(0,n-1-i):
                if arr[k][i] < arr[k + 1][i]:
                    t = arr[k]
                    arr[k] = arr[k+1]
                    arr[k+1] = t
        i = i + 1
    arr = unique(arr) #数据去重
    return arr

###一维，AC函数
###UCB AC函数
def UCB(x, Gp, kappa):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean, std = Gp.predict_f(x)
    return mean + kappa * np.sqrt(std)


###>=2维时，AC函数
###maxmin子函数
def estimate(pf,Point):
    # pf = pareto(Y_train)
    mi=0
    ma=float("-inf")
    for p in pf:
        mi = np.min(Point-p)
        if mi > ma:
            ma=mi
    if ma < 0:
        return -ma
    else:
        return 0

###maxmin AC函数，参数为X数组，及其对应的means和var数组
def mami(X,means,var):
    # AC部分
    dim = means.shape[0]  # 计算维度
    n = np.shape(X)[0]
    pf = pareto(Y_train)#######################Y_train

    Ycand = np.zeros((n, 1))
    conv = np.zeros((dim,dim))
    i=0
    while i < N:
        #print(i)
        Sum=0
        mean = np.array([means[k][i] for k in range(dim)])             # 均值
        ###方差计算
        for k in range (dim):
            conv[k][k] = var[k][i]
        Yrandom = np.random.multivariate_normal(mean=mean, cov=conv, size=1000)
        for Point in Yrandom:
            Sum =+ estimate(pf,Point)
        Ycand[i]=Sum
        i=i+1
    index=np.argmax(Ycand)
    Xcand=X[index]
    ##print(Ycand)
    ##print(Xcand)
    return Xcand
