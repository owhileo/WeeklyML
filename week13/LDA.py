from gensim import corpora,models
from gensim.parsing.preprocessing import STOPWORDS
import pandas as pd

stopw=STOPWORDS#.union(['=','-','+','•'])

def preprocess(s):
    s=s.lower().split()
    s=[i for i in s if i not in stopw]
    s=[i for i in s if i.isalpha()]
    return s


data=pd.read_csv('papers.csv')
data=data[-1000:]
paper_data=data['full_text']
paper_data=paper_data.map(lambda x:preprocess(x))

dictionary=corpora.Dictionary(paper_data)
corpus=[dictionary.doc2bow(x) for x in paper_data]

lda=models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,num_topics=100)

for topic in lda.print_topics(num_topics=10,num_words=10):
    print(topic)

# print(lda.inference(corpus))