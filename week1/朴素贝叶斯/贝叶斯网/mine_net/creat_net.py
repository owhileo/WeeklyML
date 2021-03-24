import pandas as pd
import numpy as np
import joblib


class cpt:
    def __init__(self, var, parent):
        super().__init__()
        self.var = var  # 当前变量的下标
        self.parent = parent  # 父亲节点的下标
        self.pro = {}  # 字典类型

    def add_pro(self, key, value):
        self.pro[key] = value




def k2(data, categories, max_parent):
    """
    #### 构造DAG 图
    #### 输入：
    data：整个数据集\n
    categories：数据集的列名\n
    max_parent：最大父节点数
    #### 返回：
    DAG：二维矩阵

    """
    dim = np.size(categories)
    dag = np.zeros((dim, dim))
    k2_score = np.zeros(dim)
    for i in range(dim):  # 对每一个属性
        ok = 1  # 继续添加父节点是否还会提高分数
        dim_score = -10e10  # 此属性有父节点时最高得分
        parent = np.zeros(dim)
        while(ok and np.sum(parent) < max_parent):
            local_score = -10e10
            local_node = 0
            # 因为是拓扑顺序，只与前面的节点有关
            for j in range(i-1, -1, -1):
                if(parent[j] == 0):  # 寻找不是当前父节点的节点
                    parent[j] == 1
                    tmp_score = Score(data, categories, i, parent)
                    if(tmp_score > local_score):
                        local_score = tmp_score
                        local_node = j
                    parent[j] = 0
            if(local_score > dim_score):
                dim_score = local_score
                parent[local_node] = 1
            else:
                ok = 0
        dag[i:] = parent
        k2_score[i] = dim_score
    return dag, k2_score


def ln_gamma(x):
    """
    #### 返回ln((x-1)!)
    """
    x = int(x)
    return sum(np.log(range(1, x)))



def Score(data, categories, var_num, parent):
    """
    #### 计算每一个节点和其暂时父亲节点的得分
    ### 参数：
    data: 全部数据集\n
    categories：列属性\n
    var_mun: 当前属性下标\n
    parent：DAG图中当前属性的[0,0,0,1,1,0]的值，1代表父亲节点
    ### 返回：
    score 得分
    """
    score = 0
    val = categories[var_num]  # 当前变量
    ri = np.size(val)  # 当前属性取值个数
    parent = [i for i in range(len(parent)) if parent[i] == 1]  # 返回parent属性下标
    n = np.size(data, 0)  # 有几行数据
    used = np.zeros(n)
    d = 0
    while(d < n):
        freq = np.zeros(ri)
        while(d < n and used[d] == 1):  # 找到第一个没有用过的数据
            d += 1
        if(d >= n):
            break
        for i in range(ri):
            if(val[i] == data.iloc[d, var_num]):
                break
        freq[i] = 1
        used[d] = 1
        parent_val = data.iloc[d, parent]
        d += 1
        if(d >= n):
            break
        for j in range(d, n):
            if(used[j] == 0):  # 寻找后面没有用到过的数据
                if((parent_val == data.iloc[j, parent]).all()):
                    for m in range(ri):
                        if(data.iloc[j, var_num] == val[m]):
                            break
                    freq[m] += 1
                    used[j] = 1
        sum_f = np.sum(freq)
        for j in range(ri):
            if(freq[j] != 0):
                score += ln_gamma(freq[j]+1)
        score += ln_gamma(ri)-ln_gamma(sum_f+ri)
    return score


def con_pro(data, parent_list, var_num, parent, var):
    """
    #### 根据得到的dag 可以求出相应的条件概率
    """
    sum = 0
    num = 0
    for i in range(np.size(data, 0)):  # 每一行
        if(((parent+1) == data.iloc[i, parent_list]).all()):  # 找到此数据对应父节点属性的取值
                sum += 1
                if(data.iloc[i, var_num] == (var+1)):  # 加一的原因是因为 从0开始作为数据取值（真实是从1开始的
                    num += 1
    if(sum == 0):
            return 0.001
    else:
            return round(num/sum, 3)



def class_pro(val_index, k,data): 
    """
    ### 每个类的先验概率
    ### val_index:属性的下标，k属性可以取值的个数
    ### return :类条件概率 [0.1 ,0.4 ,0.7 ]
    """
    pro={x:0 for x in range(k)}
    n=np.size(data,0)
    for i in range(n):#每行数据
        pro[ data.iloc[i,val_index]-1 ]+=1
    pro=[pro[key]/n for key in pro]
    return pro 
    



def vlaueAdd(parent_value, parent_list, parent_radix):
    """
    #### 进行加一操作（类似于二进制串）
    parent_value:父亲节点的值\n [0,0,1]
    parent_list:父亲节点的下标\n 
    parent_radix：每个父亲节点对应的取值个数[3,3,2]

    ### 返回
    parent_value +1 的list形式[0,1,0]

    """

    for i in range(np.size(parent_value)):
        if((parent_value[i]+1) % parent_radix[i] != 0):
            parent_value[i] = parent_value[i]+1
            break
        else:
            parent_value[i] = 0
    return parent_value


# # #读取数据----------------------------------------------------------------------
data = pd.read_csv("titanic.csv")
categories = data.columns.tolist()
# #用来存储每个变量有几个取值
var_range = [np.unique(data.loc[:, x]) for x in categories]


#构建DAG-----------------------------------------------------------------
# #设置最大父节点
# max_parent=5
# #设置迭代此时
# iter=1
# best_score=-10e10
# best_dag=np.zeros((1,1))
# #构建dag图
# for i in range(iter):
#     (dag,k2score)=k2(data,var_range,max_parent)
#     score=np.sum(k2score)
#     if(score>best_score):
#         best_score=score
#         best_dag=dag
# dag=dag.astype(np.int)
# joblib.dump(dag, 'dag.save')  #保存训练数据
dag = joblib.load('dag.save') #读取训练数据 
print(dag)

# # # # -------------------------cpt条件概率表获取----------------------------------
# cptList = []
# # 获取条件概率cpt 表
# for i in range(np.size(categories)):  # 对于每一个属性
#     parent_list = [x for x in range(np.size(dag[i])) if dag[i][x] == 1]
#     t_cpt = cpt(i, parent_list)
#     cptList.append(t_cpt)
#     for j in range(np.size(var_range[i])):  # 每一个属性的各种取值
#         parent_value = np.zeros(np.size(parent_list))  # 每一个父亲的取值
#         parent_radix = [np.size(var_range[x])
#                         for x in parent_list]  # [3,3,2] parent的每个节点的可取值数目
#         if(np.size(parent_radix) != 0):  # 有父亲节点的条件概率
#             for k in range(np.prod(parent_radix)):
#                 pro = con_pro(data, parent_list, i, parent_value,j)
#                 tmp = parent_value.tolist()
#                 tmp.append(j)
#                 # 以tuple（父节点取值，当前节点取值）：pro 形式的字典进行存储
#                 tmp = [int(x) for x in tmp]
#                 t_cpt.add_pro(tuple(tmp), pro)  # 即(0, 0): 0.001
#                 parent_value = vlaueAdd(
#                     parent_value, parent_list, parent_radix)

# joblib.dump(cptList, 'cptList.save')  #保存训练数据
cptList = joblib.load('cptList.save') #读取训练数据 
# print(cptList[2].pro) # cptlist 包含了所有属性的相关的条件概率 （...,0）前面几位代表其父亲节点的取值，最后一位代表当前属性的取值



#------------------------------------------------------------------------------------------------
# #获取类条件概率
# class_list=[]
# for i in range(np.size(categories)):#对于每一个属性
#     tmp=class_pro(i,np.size(var_range[i]),data)
#     class_list.append(tmp)

# joblib.dump(class_list, 'class_list.save')  #保存训练数据
class_list = joblib.load('class_list.save') #读取训练数据
# print(class_list[0])


data = pd.read_csv("titanic_test.csv")
categories = data.columns.tolist()
# #用来存储每个变量有几个取值
var_range = [np.unique(data.loc[:, x]) for x in categories]

#预测
def predict(val_index,parent_value):
    """
    ### 参数
    val_index:属性的下标\n
    parent_value:此属性的父节点的取值 (从0开始)
    ### return：
        此属性的预测值（每个可能的取值）
    """

    parent_list=cptList[val_index].parent
    pro=[]
    for i in range(np.size(var_range[val_index])):#当前属性的可能的取值
        p=parent_value[:]
        p.append(i)
        tmp_pro=max(cptList[val_index].pro[tuple(p)],0.01)*max(class_list[val_index][i],0.01)#根据求得的条件概率*类概率（可得）
        pro.append(round(tmp_pro,3))
    return pro

#真实情况：
def true_pro(val_index,parent_value):
    parent_value=[x+1 for x in parent_value]#（parent_value从1 开始）
    # print(parent_value)
    sum=0
    pro=[0 for x in range(np.size(var_range[val_index]))] #初始都是0（类的每个可能的取值）
    parent_list=cptList[val_index].parent
    for i in range(np.size(data,0)):
        if((data.iloc[i,parent_list]==parent_value).all()):
            sum+=1
            pro[data.iloc[i,val_index]-1]+=1
    return [ round(pro[x]/sum,3)  for x in range(np.size(pro))]


##预测结果-------------------------------------------------------------
# [3,3,3,3,3,2,3,2]

# pro=predict(1,[0])   
# true_pro=true_pro(1,[0]) 
# [0.563, 0.034, 0.004]
# [0.777, 0.179, 0.044]

# pro=predict(7,[1,1,1,1])
# true_pro=true_pro(7,[1,1,1,1])
# # [0.006, 0.382]
# # [0.0, 1.0]

# pro=predict(7,[0,0,1,1])
# true_pro=true_pro(7,[0,0,1,1])
# # [0.514, 0.064]
# # [0.833, 0.167]

# pro=predict(7,[0,2,0,0])
# true_pro=true_pro(7,[0,2,0,0])
# # [0.527, 0.056]
# # [0.854, 0.146]

pro=predict(7,[2,0,1,0])
true_pro=true_pro(7,[2,0,1,0])
# [0.529, 0.055]
# [0.857, 0.143]

print(pro)
print(true_pro)
