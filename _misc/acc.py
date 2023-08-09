import numpy as np
def average_every_n(vals,iters=None, n=10):
    n_excess = len(vals)%n
    n_av = len(vals)//n

    last_round_index = int(n_av*n)
    if iters is None:
        iters = np.array(range(1,1+len(vals)))
    else:
        assert(len(iters)==len(vals))
    t1 = np.array(iters[:last_round_index]).reshape(n_av,n)
    y1 = np.array(vals[:last_round_index]).reshape(n_av,n)

    t1 = list(t1[:,-1])
    y1 = list(np.mean(y1,axis=1))

    if n_excess>0:
        t1.append(iters[-1] )
        y1.append(np.mean(vals[last_round_index:]))
    return t1, y1

import os, joblib, tqdm, time
import numpy as np
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator

class MyData(Dataset):
    def __init__(self, n=2048):
        super(MyData, self).__init__()
        self.n = n
        self.D, self.C = 7, 2
        self.data = np.random.normal(0,1,size=(self.n, self.D,))

    def __getitem__(self, i):
        return self.data[i], int(np.sum(self.data[i]) > 0)

    def __len__(self):
        return self.n

class MLP(nn.Module):
    def __init__(self, D, C):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(D, 5)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(5, C)

    def forward(self,x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


def train():
    verbose = 100
    num_workers = 0

    accelerator = Accelerator()
    print('accelerator.device:',accelerator.device) # accelerator.device: cuda
    
    mydataset = MyData(n=2048)
    trainloader = DataLoader(mydataset, batch_size=16, shuffle=True, num_workers=num_workers)
    valdataset = MyData(n=2048)
    valloader = DataLoader(valdataset, batch_size=16, num_workers=num_workers)

    model = MLP(mydataset.D, mydataset.C)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5,0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
        lr_lambda=lambda epoch: 0.65 ** epoch)
    model, optimizer, trainloader, valloader, scheduler = accelerator.prepare(
        model, optimizer, trainloader, valloader, scheduler)

    criterion = nn.CrossEntropyLoss()

    start = time.time()

    lrs, losses, val_losses = [], [], []
    disable_tqdm=True if verbose==0 else False
    for ep in range(8):
        progress = tqdm.tqdm(enumerate(trainloader), total=len(trainloader),
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', # bar size
            disable=disable_tqdm,)

        model.train()
        for i,(x,y0) in progress:
            if ep==0 and i<4:
                progress.write(f"{i}:{x.shape}{y0}")
            progress.set_description('[epoch %-3s]'%(str(ep)))

            optimizer.zero_grad()
            y = model(x.to(torch.float))
            loss = criterion(y,y0)
            accelerator.backward(loss)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

        model.eval()
        for i, (x,y0) in enumerate(valloader):
            y = model(x.to(torch.float))
            loss = criterion(y,y0)     
            val_losses.append(loss.item())       

    end = time.time()
    elapsed = end - start
    print(' time taken %s[s] = %s [min] = %s [hr]'%(str(round(elapsed,1)), 
        str(round(elapsed/60.,1)), str(round(elapsed/3600.,1))))


    iters, losses1 = average_every_n(losses, iters=np.arange(len(losses)), n=64)
    val_iters, val_losses1 = average_every_n(val_losses, iters=np.arange(len(val_losses)), n=64)
    plt.figure(figsize=(4,7))
    plt.gcf().add_subplot(211)
    plt.gca().plot(lrs)
    plt.gca().set_xlabel('learning rate')
    plt.gcf().add_subplot(212)
    plt.gca().plot(losses, alpha=0.27)
    plt.gca().plot(iters, losses1, label='train')
    plt.gca().plot(val_iters, val_losses1, label='val')
    plt.gca().set_xlabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train()