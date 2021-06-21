import os
import random
import librosa
import numpy as np
import librosa.display
from hmmlearn import hmm
import pickle


np.random.seed(42)
file_name_path = "./free-spoken-digit-dataset-master/recordings/"
file_name = os.listdir(file_name_path)
file_name.sort()
n = int(len(file_name) / 10)
file_list = [file_name[i:i + n] for i in range(0, len(file_name), n)]


def load_data(category):
    files = file_list[category]
    np.random.shuffle(files)
    datas = []
    lengths = []
    for file in files:
        y, sr = librosa.load(file_name_path + file)
        ret = librosa.feature.mfcc(y, sr=sr)
        # print(ret.shape)
        datas.append(ret.T)
        lengths.append(ret.shape[1])

    train_num = int(len(files) * 0.7)
    train_data = datas[:train_num]
    test_data = datas[train_num:]
    train_lengths = lengths[:train_num]
    test_lengths = lengths[train_num:]
    return train_data, test_data, train_lengths, test_lengths

def train_GaussianHMM(train_datas, train_lengths, n_com):
    train_data = np.concatenate(train_datas)
    model = hmm.GaussianHMM(n_components=n_com)
    model.fit(train_data, train_lengths)
    return model

# def train_GMMHMM(train_datas, train_lengths):
#     train_data = np.concatenate(train_datas)
#     model = hmm.GMMHMM(n_components=10, n_mix=4)
#     model.fit(train_data, train_lengths)
#     return model

def test(models, test_datas, test_lenghts):
    total = 0
    true = 0
    for category in range(10):
        test_data = test_datas[category]
        for t in test_data:
            total += 1
            scores = []
            for model in models:
                scores.append(model.score(t))
            predict = scores.index(max(scores))
            # print(scores)
            if predict == category:
                true += 1
            else:
                pass
    print(true, total, true / total)


def main():
    train_datas = []
    train_lengths = []
    test_datas = []
    test_lengths = []
    if os.path.exists('data.pkl'):
        with open('data.pkl', 'rb') as file:
            train_datas, train_lengths, test_datas, test_lengths = pickle.load(file)
    else:
        for category in range(10):
            train_data, test_data, train_length, test_length = load_data(category)
            train_datas.append(train_data)
            test_datas.append(test_data)
            test_lengths.append(test_length)
            train_lengths.append(train_length)
        with open('data.pkl', 'wb') as file:
            pickle.dump([train_datas, train_lengths, test_datas, test_lengths], file)
    print('load OK')
    for i in range(1, 41):
        models = []
        for category in range(10):
            model = train_GaussianHMM(train_datas[category], train_lengths[category], i)
            models.append(model)
        test(models, test_datas, test_lengths)


    # models = []
    # for category in range(10):
    #     model = train_GMMHMM(train_datas[category], train_lengths[category])
    #     models.append(model)
    # test(models, test_datas, test_lengths)

    # models = []
    # for category in range(10):
    #     model = train_new(train_datas[category], train_lengths[category])
    #     models.append(model)
    # test(models, test_datas, test_lengths)

main()
