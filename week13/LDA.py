from gensim import corpora,models,matutils
from gensim.parsing.preprocessing import STOPWORDS
import pandas as pd
import scipy.sparse
import numpy as np
import random

stopw=STOPWORDS#.union(['=','-','+','•'])

class LDA(object):
    def __init__(self,corpus,id2word,num_topics,alpha=None,beta=None) -> None:
        super().__init__()
        self.corpus=corpus
        self.id2word=id2word
        self.n_topics=num_topics
        self.n_doc,self.n_word=corpus.get_shape()
        if alpha==None:
            self.alpha=np.ones(self.n_topics)
        else:
            self.alpha=alpha
        if beta==None:
            self.beta=np.ones(self.n_word)
        else:
            self.beta=beta
        self.Z=np.random.randint(low=0,high=self.n_topics,size=(self.n_doc,self.n_word))
        self.distribution=np.zeros((self.n_topics,self.n_word))

    def train_one(self,record=False):
        new_Z=np.zeros((self.n_doc,self.n_word),dtype=int)
        n_kd=np.zeros((self.n_doc,self.n_topics))
        n_tk=np.zeros((self.n_topics,self.n_word))        
        for i in range(self.n_doc):
            for j in range(self.n_word):
                if self.corpus[i,j]>0:
                    topic=self.Z[i,j]
                    n_kd[i,topic]+=self.corpus[i,j]
                    n_tk[topic,j]+=self.corpus[i,j]
                    if record:
                        self.distribution[topic,j]+=self.corpus[i,j]
        count=0
        for d in range(self.n_doc):
            for i in range(self.n_word):
                if self.corpus[i,j]==0:
                    continue
                t=np.zeros(self.n_topics)
                tt=np.zeros(self.n_topics)
                for k in range(self.n_topics):
                    n_kdi=n_kd[d,k]+self.alpha[k]
                    n_tki=n_tk[k,i]+self.beta[i]
                    n_tki_sum=np.sum(self.beta)+np.sum(n_tk[k,:])
                    if self.Z[d,i]==k:
                        n_kdi-=self.corpus[d,i]
                        n_tki-=self.corpus[d,i]
                        n_tki_sum-=self.corpus[d,i]
                        
                    t[k]=n_kdi
                    if n_tki==0:
                        tt[k]=0
                    else:
                        tt[k]=n_tki/n_tki_sum

                t/=np.sum(t)
                res=t*tt
                res/=np.sum(res)
                r=random.random()
                r_=0
                ix=0
                for ix,x in enumerate(res):
                    r_+=x
                    if r<x:
                        break
                new_Z[d,i]=ix
                if ix!=self.Z[d,i]:
                    count+=1
        self.Z=new_Z
        return count

    def train(self,n=20):
        for _ in range(3):
            res=self.train_one()
        for i in range(n):
            res=self.train_one(True)
            print(res)
        
        for topic in range(self.n_topics):
            self.distribution[topic,:]/=np.sum(self.distribution[topic,:])

    def print_topics(self,n_word=10):
        for k in range(self.n_topics):
            print("topic %d:"%k)
            t=[(x,y) for x,y in enumerate(self.distribution[k,:])]
            t=sorted(t,key=lambda x:x[1])
            res=[(y,self.id2word[x]) for x,y in t]
            print(res[:n_word])
    

def preprocess(s):
    s=s.lower().split()
    s=[i for i in s if i not in stopw]
    s=[i for i in s if i.isalpha()]
    return s

if __name__=='__main__':

    data=pd.read_csv('papers.csv',nrows=20)
    #data=data[-10:]
    paper_data=data['full_text']
    paper_data=paper_data.map(lambda x:preprocess(x))

    dictionary=corpora.Dictionary(paper_data)
    corpus=[dictionary.doc2bow(x) for x in paper_data]
    corpus=matutils.corpus2csc(corpus)
    lda=LDA(corpus=corpus,id2word=dictionary,num_topics=5)
    lda.train()
    lda.print_topics()

# lda=models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,num_topics=100)

# for topic in lda.print_topics(num_topics=10,num_words=10):
#     print(topic)



# print(lda.inference(corpus))