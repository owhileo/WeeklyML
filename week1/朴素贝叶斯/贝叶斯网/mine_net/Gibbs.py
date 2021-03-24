import random
import numpy as np
class bayesNode:
    def __init__(self,x,parent,cpt):
        self.x=x
        self.parent=parent 
        self.cpt=cpt   #没有父亲的时候就是一个数字
        self.child=[]

        
    
    def pro(self,event,x_val):
        tmp=[x_val]
        if(self.parent!=''):
            for val in self.parent:
                tmp.append(event[val])
        tmp=tuple(tmp)
        p=self.cpt[tmp]
        return p
        


class bayesNet:
    def __init__(self,node_specs):
        self.nodes=[]
        self.variable=[]
        for var in node_specs:
            node=bayesNode(*var)
            self.variable.append(node.x)
            self.nodes.append(node)
            
            for parent in node.parent:
                if(parent!=''):
                    self.tran2Node(parent).child.append(node)

    

    def tran2Node(self,val):#将val 转化成node
        for node in self.nodes:
            if(node.x==val):
                return node

#gibbs 预测
def Gibbs(x,val_num,evidc,BN,n):#n是迭代的次数 ，x_num 是x可以取值的个数
    count={i:0 for i in range(val_num[x])}  #每个类取值从0开势
    Z=[var for var in BN.variable if var not in evidc] #非证据因子
    state=evidc
    for Zi in Z:
        state[Zi]=random.choice(range(val_num[Zi]))  #写成一个字典才好处理
    for j in range(n):#迭代的次数
        for zi in Z:#每一个可以随机游走的属性
            T=[]
            for num in range(val_num[zi]):#属性zi的可取的值
                state[zi]=num
                t=BN.tran2Node(zi).pro(state,num)#计算zi和当前状态下zi父节点条件概率
                for y in BN.tran2Node(zi).child:
                    t*=y.pro(state,state[y.x])#计算zi的孩子节点和zi的条件概率
                T.append(t)
            state[zi]=probability(T)#按照一定的分布去根据其概率进行选择
        count[state[x]]+=1
    for key in count:
        count[key]=count[key]/n
    return  count
    

def probability(p):  #以一定的概率去选择去采样
    for i in range(len(p)):
        p[i]=p[i]/np.sum(p)
    for i in range(len(p)):
        if(p[i] > random.uniform(0.0, 1.0)):
            return i
    return random.choice(range(len(p)))
        


#我的输入
BN = bayesNet([('A','',{(0,):0.1, (1,):0.3 ,(2,):0.6} ),('B','', {(0,):0.7, (1,):0.3 }),
                ('C','A',{(0,0):0.5,(0,1):0.3,(0,2):0.2,(1,0):0.2,(1,1):0.3,(1,2):0.5}),
                ('D','BC',{(0,0,0):0.1,(0,0,1):0.3,(0,1,0):0.5,(0,1,1):0.7,(1,0,0):0.1,(1,0,1):0.3,(1,1,0):0.8,(1,1,1):0.7}),
                ('E','D',{(0,0):0.3,(0,1):0.5,(1,0):0.7,(1,1):0.5,(2,0):0.2,(2,1):0.8})
                ])
# ans=Gibbs('D',{'A':3,'B':2,'C':2,'D':2},{'B':1,'C':1},BN,1000)
ans=Gibbs('E',{'A':3,'B':2,'C':2,'D':2,'E':3},{'D':1},BN,1000)
# {0: 0.366, 1: 0.272, 2: 0.362}
print(ans)