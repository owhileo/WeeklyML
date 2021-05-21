
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


####计算二维帕累托前沿未支配的区域面积,Hv的子函数
def TwoDCalculate(arr,edges):
    area = 0.0
    arr = bubble_sort(arr,1)
    arr = Append(arr, edges[0][1], edges[1][1])
    arr = bubble_sort(arr,1)
    # print(arr)

    n = len(arr)
    for i in range(0, n):
        if i % 2 == 0:
            # print(area)
            if i == 0:
                area = area + (arr[0][0] - edges[0][0]) * (arr[0][1] - edges[1][0])
            else:
                area = area + (arr[i][0] - arr[i - 1][0]) * (arr[i][1] - edges[1][0])
        else:
            continue
    return area

####计算>2维时帕累托前沿的未支配区域的超体积,Hv的子函数
def HighDCalculate(arr,edges,volume = 0):
    dim = arr.shape[0]
    arr = pareto(Y)#################获取pareto前沿
    arr = bubble_sort(arr)
    n = len(arr)

    if dim ==2:
        return TwoDCalculate(arr,edges)

    for i in range(1,n):
        num = 1
        for j in range(i+1,n):
            if arr[j][dim-1] < arr[i][dim-1]: break
            num = num + 1

        subarr = arr[i:i+num,0:dim-1]
        subedges = edges[0:dim-1]
        #根据pareto前沿特征，以高低层中低层的dim-1维超体积 * dim维度上的高低层高差（计算未支配区域超体积）
        if i != 1:
            volume = volume = (arr[i-1][dim-1]-arr[i][dim-1])*HighDCalculate(subarr,subedges)######递归点
        else: ###dim维度上最高层
            volume = (edges[dim - 1][1] - arr[i][dim - 1]) * HighDCalculate(subarr, subedges)  ######递归点

        i = i + num
        ###dim维度上的最底层
        if i - 1 == n - 1:
            volume =+ (arr[i][dim - 1] - edges[dim - 1][0]) * np.prod(edges[0:dim - 1, 1] - edges[0:dim - 1, 0])
            return volume

    ##三维计算
    # for i in range(1, N):
    #     flag = 0
    #     xy = np.array([[0.0, 0.0]])
    #     ###xy选择时需考虑下层点
    #     if z[i] == -39.8:
    #         area = (x[1] - x[0]) * (y[1] - y[0])
    #         print(area)
    #         Volume = Volume + area * (z[i - 1] - z[i])
    #         Volume = (x[1] - x[0]) * (y[1] - y[0]) * (-38.9 + 39.8) - Volume
    #         print(Volume)
    #         break
    #     for j in range(0, n):
    #         if Z[j] <= z[i]:
    #             if flag == 0:
    #                 xy[0] = XY[j]
    #                 flag = 1
    #             else:
    #                 xy = np.concatenate((xy, [XY[j]]))
    #     print(xy)
    #     area = TwoDCalculate(xy, x, y)
    #     Volume = Volume + area * (z[i - 1] - z[i])
    #     print(Volume)
