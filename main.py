import json
import torch 
import torch.nn as nn
from model import TinyGPT
from tqdm import tqdm
import pandas as pd 
import time 
from data.utils import TextDataset
from torch.utils.data import DataLoader


DEVICE='mps' if torch.mps.is_available() else 'cpu'
with open('data/vocab.json','r',encoding='utf-8') as f:
    vocab=json.load(f)

encode=lambda s: [vocab[c] for c in s]
decode=lambda l: "".join([k for id in l for k,v in vocab.items() if id==v])

with open("data/input.txt","r",encoding="utf-8") as f:
    text=f.read()

block_size=64
data=torch.tensor(encode(text),dtype=torch.long,device=DEVICE)
n=data.size(0)// block_size*block_size 
data=data[:n].view(-1,block_size)
chars=sorted(list(set(text)))


dataset=TextDataset(data)
dataloader=DataLoader(dataset,batch_size=64,shuffle=True)

model=TinyGPT(vocab_size=len(chars),block_size=block_size,n_emb=128)
model.to(DEVICE)
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn=nn.CrossEntropyLoss()
checkpoint={}
track_loss=[]
training_time=[]
for epoch in tqdm(range(1,11)):
    start=time.perf_counter()
    for x,y in dataloader:
        optimizer.zero_grad()
        logits=model(x)
        loss=loss_fn(logits.view(-1,logits.size(-1)),y.view(-1))
        loss.backward()
        optimizer.step()
        
        checkpoint={
            "epoch":epoch,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "loss":loss
        }
    end=time.perf_counter()
    print(f"Epoch {epoch} Loss {loss.item()}")
    track_loss.append(loss)
    training_time.append(end-start)




#torch.save(model.state_dict(),"")
pd.DataFrame({"Epoch":[num+1 for num in range(len(track_loss))],"Loss":track_loss,"Time":training_time}).to_csv('Training_metadata.csv')

torch.save(checkpoint,f"checkpoint_{epoch}.pth")