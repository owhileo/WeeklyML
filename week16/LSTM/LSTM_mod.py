# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 21:53:07 2021
@author: nijie
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 21:35:06 2021
@author: nijie
"""
import torch
from torch import nn
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import time
from sklearn import metrics
from tqdm import tqdm
from collections import Counter
import torch.nn.functional as F

#from google.colab import drive
#drive.mount('/content/gdrive')
#os.chdir('/home/lamda10/Nick')
import os
os.chdir('C:/py_doc')

#提取&清洗
filedir = 'train.csv'
filedir_test = 'test.csv'

train = pd.read_csv(filedir, header = None)
train = train.drop(0, axis = 1)
train.columns = ['des', 'label']
test = pd.read_csv(filedir_test, header = None)
test.columns = ['ID', 'des']
train_backup=train
test_backup=test
# # 暂时设置
train= train[:50]
test= test[:50]

train['des'] = train.apply(lambda x: x['des'].strip('|').strip(), axis=1)
train['label'] = train.apply(lambda x: x['label'].strip('|').strip(), axis=1)
test['des'] = test.apply(lambda x:x['des'].strip('|').strip(), axis=1)

train_des = []
for des in np.array(train['des']):
    train_des.append([int(i) for i in des.split()])
train_lab = []
for lab in np.array(train['label']):
    train_lab.append([int(i) for i in lab.split()])
test_des = []
for des in np.array(test['des']):
    test_des.append([int(i) for i in des.split()])
lenlist=[]
for i in range(len(train_des)):
    lenlist.append(len(train_des[i]))
print(min(lenlist))
print(max(lenlist))




# 把所有的单词编码
# 所有的单词拉成一个list
all_words = train_des[0]
for i in range(1,len(train_des)):
    all_words = all_words+train_des[i]
for i in range(len(test_des)):
    all_words = all_words+test_des[i]

vocab_dict = dict(Counter(all_words))#计数

word2idx = {w:i for i, w in enumerate(list(set(vocab_dict.keys())))} #一个字典，w：字(key)，i：索引(value)
idx2word = {i:w for i, w in enumerate(list(set(vocab_dict.keys())))} #一个字典，w：字(value)，i：索引(key) #取消
word2idx['<UNK>'] = len(word2idx)

word_counts = np.array([count for count in vocab_dict.values()], dtype=np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3./4.)

print("字典长度,", "训练+测试集中字的个数+<UNK>,", "vocab_size,", len(word2idx)) # 字典长度


# 所有类别的索引
# 所有类别拉成一个list
all_labels = train_lab[0]
for i in range(1, len(train_lab)):
    if len(train_lab[i]) == 0:
        all_labels = all_labels+[-1] #“+”两个list合并在一起
    else:
        all_labels = all_labels+train_lab[i]

label2idx = {l:i for i , l in enumerate(list(set(all_labels)))} # 一个字典，l:类别(key)， i:索引(value)
idx2label = {i:l for i, l in enumerate(list(set(all_labels)))}  # 一个字典，l:类别(value)， i:索引(key) #去掉不连续的标签
lab_cnt = len(label2idx)
print("类别个数,", "训练集中类别个数,", "vocab_size,", lab_cnt) #类别个数


class WordEmbeddingDataset(Dataset):
    def __init__(self, text, word2idx, word_freqs):
        ''' text: a list of words, all text from the training dataset
            word2idx: the dictionary from word to index
            word_freqs: the frequency of each word
        '''
        super(WordEmbeddingDataset, self).__init__() # #通过父类初始化模型，然后重写两个方法
        self.text_encoded = [word2idx.get(word, word2idx['<UNK>']) for word in text] # 把单词数字化表示。如果不在词典中，也表示为unk
        self.text_encoded = torch.LongTensor(self.text_encoded) # nn.Embedding需要传入LongTensor类型
        self.word2idx = word2idx
        self.word_freqs = torch.Tensor(word_freqs)
        
    def __len__(self):#__重载__
        return len(self.text_encoded) # 返回所有单词的总数，即item的总数
    
    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的positive word
            - 随机采样的K个单词作为negative word
        '''
        center_words = self.text_encoded[idx] # 取得中心词
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1)) # 先取得中心左右各C个词的索引
        pos_indices = [i % len(self.text_encoded) for i in pos_indices] # 为了避免索引越界，所以进行取余处理
        pos_words = self.text_encoded[pos_indices] # tensor(list)
        
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)
        # torch.multinomial作用是对self.word_freqs做K * pos_words.shape[0]次取值，输出的是self.word_freqs对应的下标
        # 取样方式采用有放回的采样，并且self.word_freqs数值越大，取样概率越大
        # 每采样一个正确的单词(positive word)，就采样K个错误的单词(negative word)，pos_words.shape[0]是正确单词数量
        
        # while 循环是为了保证 neg_words中不能包含背景词
        while len(set(pos_indices) & set(neg_words)) > 0:
            neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)

        return center_words, pos_words, neg_words
    

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
         
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)
        
    def forward(self, input_labels, pos_labels, neg_labels):
        ''' input_labels: center words, [batch_size]
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels：negative words, [batch_size, (window_size * 2 * K)]
            
            return: loss, [batch_size]
        '''
        input_embedding = self.in_embed(input_labels) # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels)# [batch_size, (window * 2), embed_size]
        neg_embedding = self.out_embed(neg_labels) # [batch_size, (window * 2 * K), embed_size]
        
        input_embedding = input_embedding.unsqueeze(2) # [batch_size, embed_size, 1]
        
        pos_dot = torch.bmm(pos_embedding, input_embedding) # [batch_size, (window * 2), 1]
        pos_dot = pos_dot.squeeze(2) # [batch_size, (window * 2)]
        
        neg_dot = torch.bmm(neg_embedding, -input_embedding) # [batch_size, (window * 2 * K), 1]
        neg_dot = neg_dot.squeeze(2) # batch_size, (window * 2 * K)]
        
        log_pos = F.logsigmoid(pos_dot).sum(1) # .sum()结果只为一个数，.sum(1)结果是一维的张量
        log_neg = F.logsigmoid(neg_dot).sum(1)
        
        loss = log_pos + log_neg
        
        return -loss
    
    def input_embedding(self):#传出当前训练好的权重
        return self.in_embed.weight.detach().numpy()
    
    
C = 3
K = 15
batch_size = 256
dataset = WordEmbeddingDataset(all_words, word2idx, word_freqs)
dataloader = DataLoader(dataset, batch_size, shuffle=True)
model = EmbeddingModel(860, 300)# vocab_size = 859, embedding_size = 300
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for e in range(10):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()
        
        optimizer.zero_grad()#梯度初始化为0
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        
        optimizer.step()
        
        if i % 100 == 0:
            print('epoch', e, 'iteration', i, loss.item())
            
embedding_weights = model.input_embedding()
torch.save(model.state_dict(), "embedding-{}.th".format(300))


class textDataset(Dataset):
    def __init__(self, df, idx):
        super().__init__()
        df = df.loc[idx,:].reset_index(drop=True)
        self.text_lists = df['des'].values
        self.labels = df['label'].values
    def get_dumm(self,s):
        re=[0]*17
        if s=='':
            return re
        else:
            tmp=[int(i) for i in s.split(' ')]
            for i in tmp:
                re[i]=1
        return re
    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self,idx):
        text = self.text_lists[idx]
#         print(text)
        text = [word2idx[int(i)] for i in text.split()]
        if len(text)>100:
            text = text[:100]
        else:
            text = text+[859]*(100-len(text))
        
        text_vec = np.array([embedding_weights[i] for i in text])
        # print(text_vec.shape)
        label = self.labels[idx]
        label=self.get_dumm(label)
#         print("text:", text)
#         print('-----------in textDataset')
        return torch.Tensor(text_vec), np.array(label)
    
    
class textLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(textLSTM,self).__init__()
        # vocab_size: 所有词的个数, embedding_dim: 词嵌入向量维数
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # self.center_embed = nn.Embedding(vocab_size, embedding_dim)
        # self.context_embed = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, bias = True,
                            num_layers = 2, bidirectional = True, batch_first = True, dropout=0.2)
        self.lstm_out = None
        self.fc = nn.Linear(in_features = hidden_dim*2, out_features = 17)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
      # x.shape = [batch_size, text_len] 其中text_len是每个句子长度(初始100截断，见Dataset)
#         print(x)
        # print(x.shape)
        # embedding = self.embedding(x)
        # print('embedding', embedding.shape, embedding.type)
        self.lstm_out, (hidden, cell) = self.lstm(x)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
#         print(hidden)
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        return out
    

vocab_size = 858+2 # 所有训练文本的单词个数
embedding_dim = 300 # 词嵌入维数
hidden_dim = 200
max_epoch = 10
print_interval=-1
criterion = torch.nn.BCEWithLogitsLoss()

net = textLSTM(vocab_size, embedding_dim, hidden_dim)

optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=5e-4) # lr:learning rate, weight_decay:权重递减
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2, eta_min=1e-5, last_epoch=-1)
#调整学习率
train_data = textDataset(train, list(range(len(train))))
train_loader = DataLoader(train_data, batch_size = 256, shuffle = True)
val_data = textDataset(train, list(range(len(train)-10, len(train))))
#改一改
val_loader = DataLoader(val_data)


@torch.no_grad()
def val_model(model, criterion):
    model.eval()
    pres_list=[]
    labels_list=[]
    for i,(inputs, labels) in enumerate(val_loader):
        inputs = inputs.type(torch.Tensor)
        labels = labels.type(torch.Tensor)
        outputs = model(inputs)

        pres_list+=outputs.sigmoid().detach().cpu().numpy().tolist()
        labels_list+=labels.detach().cpu().numpy().tolist()
    #val_auc = metrics.roc_auc_score(labels_list, pres_list, multi_class='ovo')
    log_loss=metrics.log_loss(labels_list, pres_list)
    print('valLogLoss: {:.4f}'.format(log_loss))#保留4位小数
    return log_loss

iter_cnt = len(train_loader)
print('total iters:{}'.format(iter_cnt))

iters = len(train_loader)
since = time.time()
print("开始训练时间=",since)

# 记录最优的epoch和loss
best_loss = 1e7
best_auc = 0
best_epoch = 0

for epoch in range(1, max_epoch+1):
    net.train(True)
    begin_time = time.time()
    print('learning rate:{}'.format(optimizer.param_groups[-1]['lr']))
    print('Epoch{}/{}'.format(epoch, max_epoch))
    print('-'*20)
    count = 0
    train_loss = []

    size = [0, 100, 400]
    # lstm的训练词向量结果
    lstm_output = torch.empty(size)
    print(1)
    for i, (inputs, labels) in enumerate(train_loader):
        count = count+1
#         print(inputs)
        inputs = torch.Tensor(inputs)
        # print(inputs.shape)
        labels = labels.float()
        out_linear = net(inputs)
        loss = criterion(out_linear, labels)
        optimizer.zero_grad()
        loss.backward() #误差的反向传播，tensor，知道loss进行过的数学运算就能自动梯度
        optimizer.step()#更新参数
        lstm_output = torch.cat([lstm_output, net.lstm_out], 0)
        #print(lstm_output.shape)
    print(3)
    print(lstm_output.detach().numpy().shape)  
    lr_scheduler.step()
    #model_save_dir = '/content'
#     print('lr = ',optimizer.param_groups[-1]['lr'])
    val_loss= val_model(net, criterion)
    best_model_out_path ='epoch_'+str(epoch)+'_best'+'.pth'
    #save the best model
    print(4)
    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch=epoch
        torch.save(net.state_dict(), best_model_out_path)
        np.save('epoch_'+str(epoch)+'_best_wordvec.npy',lstm_output.detach().numpy())
        print("save best epoch: {} best logloss: {}".format(best_epoch,val_loss))
    #save based on epoch interval
    #if epoch % 5  == 0 and epoch>30:
        #torch.save(model.state_dict(), model_out_path)
#
    print('Best logloss: {:.3f} Best epoch:{}'.format(best_loss,best_epoch))
    time_elapsed = time.time() - since
    print('loss = ', loss.item())
    
    
def load_model(weight_path):
    print(weight_path)
    model=textLSTM(vocab_size, embedding_dim, hidden_dim)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    return model

model = load_model('epoch_10_best.pth')

model.eval()
pres_list=[]
labels_list=[]
for i,(inputs, labels) in enumerate(val_loader):
    inputs = inputs.type(torch.Tensor)
    labels = labels.type(torch.Tensor)
    outputs = model(inputs)

    pres_list+=outputs.sigmoid().detach().cpu().numpy().tolist()
    labels_list+=labels.detach().cpu().numpy().tolist()
pres = np.array(pres_list)
labels = np.array(labels_list)
temp_loss = 0
for i in range(pres.shape[0]):
    for j in range(pres.shape[1]):
        temp_loss = temp_loss+labels[i][j]*np.log(pres[i][j])+(1-labels[i][j])*np.log(1-pres[i][j])
temp_loss = -temp_loss/(pres.shape[0]*pres.shape[1])
temp_loss = 1-temp_loss
print(temp_loss)


pres_all=[]
texts = np.array(test_des)
for text in tqdm(texts):
    if len(text)>100:
        text=text[:100]
    else:
        text=text+[859]*(100-len(text))
    text_vec = np.array([[embedding_weights[i] for i in text]])
    # text=torch.from_numpy(np.array(text))
    # text=text.unsqueeze(0)
    text_vec=torch.Tensor(text_vec)#.cuda()
    #
    outputs=model(text_vec)
    
    pres_fold=outputs.sigmoid().detach().cpu().numpy()[0]
    
    pres_fold=[str(p) for p in pres_fold]
    pres_fold=' '.join(pres_fold)
    pres_all.append(pres_fold)
    
    
sub_id=test['ID'].values
save_dir = 'C:/py_doc/'
if not os.path.exists(save_dir): os.makedirs(save_dir)
str_w=''
with open(save_dir+'submit.csv','w') as f:
    for i in range(len(sub_id)):
#         print(i)
#         print(sub_id[i], pres_all[i])
        str_w+=sub_id[i]+','+'|'+pres_all[i]+'\n'
    str_w=str_w.strip('\n')
    f.write(str_w)
