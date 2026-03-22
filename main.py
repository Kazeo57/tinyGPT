import torch 
import torch.nn as nn
from model import TinyGPT

from data import TextDataset
from torch.utils.data import DataLoader


dataset=TextDataset(data)
dataloader=DataLoader(dataset,batch_size=64,shuffle=True)

model=TinyGPT(vocab_size=len(chars),block_size=block_size,n_emb=128)
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn=nn.CrossEntropyLoss()

for epoch in range(10):
    for x,y in dataloader:
        optimizer.zero_grad()
        logits=model(x)
        loss=loss_fn(logits.view(-1,logits.size(-1)),y.view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} Loss {loss.item()}")