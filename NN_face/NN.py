import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import pickle
import re

dtype = torch.float32
torch.backends.cudnn.benchmark = True


def sorted_aphanumeric(data):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]

    return sorted(data, key=alphanum_key)


def save():
    data = MyDataset()
    fw = open('../DataLoader', 'wb')
    pickle.dump(data, fw)
    fw.close()


def load():
    try:
        fr = open('../DataLoader', 'rb')
    except:
        fr = open('DataLoader', 'rb')
    data = pickle.load(fr)
    fr.close()
    return data


def hash37(value):
    hsh = int(21390)
    hsh = value + 37 * hsh

    return hsh


batch_size = 3000


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        #self.l0 = nn.BatchNorm1d(464*3)
        self.l1 = nn.Linear(4, 8)
        self.l2 = nn.Linear(8, 16)
        self.l3 = nn.Linear(16, 32)
        self.l5 = nn.Linear(32, 32)
        self.l6 = nn.Linear(32, 64)
        self.l7 = nn.Linear(64, 50)

    def forward(self, x):
        model = torch.nn.Sequential(
            #self.l0,
            self.l1,
            nn.ReLU(),
            self.l2,
            nn.ReLU(),
            self.l3,
            nn.ReLU(),            
            self.l5,
            nn.ReLU(),
            self.l6,
            nn.ReLU(),
            self.l7
        )
        return model(x)


class MyDataset(Dataset):

    def __init__(self):
        self.X = np.array([])
        self.y = np.array([])
        os.chdir('train_data/abc')
        data = sorted_aphanumeric(os.listdir())
        for k, i in enumerate(data):
            df1 = pd.read_csv('%s' % (i))
            if k == 0:
                self.X = np.append(self.X, [np.array([1, 0, 0, 1])])
            elif k == 1:
                self.X = np.append(self.X, [np.array([0, 1, 1, 0])])
            if len(df1) < 25:
                for i in range(25-len(df1)):
                    df1 = df1.append({'rule_1':0, 'rule_2':0}, ignore_index=True)
            if len(self.y) == 0:
                self.y = [df1.to_numpy()]
            else:
                self.y = np.vstack((self.y, [df1.to_numpy()]))

        self.y = self.y.reshape((2,50))
        self.X = self.X.reshape((2,4))
        self.X = torch.FloatTensor(self.X).cuda()
        self.y = torch.FloatTensor(self.y).cuda()

        os.chdir('../../')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return (self.X[index], self.y[index])


def main():
    lr = 1e-4
    net = NN()
    net = net.to('cuda')
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    data = load()
    loss_f = nn.MSELoss().cuda()
    #transform = torch.Tensor([1 for i in range(1002)]).cuda()

    train = [DataLoader(data.X.to('cuda'), batch_size=batch_size),
             DataLoader(data.y.to('cuda'), batch_size=batch_size)]
    valid = [DataLoader(data.X[int(len(data.X)*0.8):].to('cuda'), batch_size=batch_size),
             DataLoader(data.y[int(len(data.X)*0.8):].to('cuda'), batch_size=batch_size)]
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2_000_000, gamma=0.5)

    tmp1 = zip(*train)
    tmp2 = zip(*valid)
    tr = list(tmp1)
    vl = list(tmp2)

    for i in tqdm(range(1_000_000)):
        net.train()
        for x, y in tr:
            optimizer.zero_grad()
            predicted_values = net(x)
            loss = loss_f(torch.abs(predicted_values-y).sum(),(y-y).sum())
            loss.backward()
            optimizer.step()
        scheduler.step()
        #net.eval()
        #with torch.no_grad():
        #    for x, y in vl:
        #        predicted_values = net(x)
        #        loss = loss_f(predicted_values, y)
        if i % 50_000 == 0 and i > 0:
            print('\n')
            print(loss.item())

    savePolicy(net)


def savePolicy(model):
    torch.save(model, 'abc.pth')


def loadPolicy(name):

    model = torch.load('%s.pth' % (name))
    model.eval()
    return model


def test():
    #trans = torch.FloatTensor([1 for i in range(1002)]).cuda()
    net_right = loadPolicy('abc')
    #net_left = loadPolicy('left_best')
    #df1 = pd.read_csv('train_data/abc/pos1.csv.csv')
    #df2 = pd.read_csv('train_data/left/100.0_1.csv')
    #train1 = df1.iloc[:, 1:4].to_numpy()
    #train2 = df2.iloc[:, 1:4].to_numpy()
    #data1 = train1.reshape((1, train1.shape[0] * train1.shape[1]))
    #data2 = train2.reshape(train2.shape[0] * train2.shape[1])
    #data = torch.Tensor(train).to('cuda')
    ot1 = net_right(torch.Tensor([0, 1, 1, 0]).cuda())
    #ot2 = net_left(torch.Tensor(data2).cuda())
    #out = torch.matmul(ot.T, trans.T).cuda()

    df1 = pd.DataFrame(ot1.reshape((25,2)).cpu().detach().numpy(), columns=['rule1', 'rule2'])
    df = pd.read_csv(r'D:\work_Denis\NN_face\train_data\abc\pos2.csv')
    df1[df1 < 0.001] = 0
    df1 = df1.join(df)
    df1.to_csv(r"D:\work_Denis\NN_face\train_data\abc\out_NN.csv" , index=False)
    # print(ot2)


#save()
#main()
test()
