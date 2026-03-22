import torch
import subprocess

data=torch.tensor(encode(text),dtype=torch.long)
n=data.size(0)// block_size**2 
data=data[:n].view(-1,block_size)

#dataLoader 
from torch.utils.data import Dataset 

subprocess.run("wget https://www.gutenberg.org/files/1342/1342-0.txt -O input.txt")
with open("input.txt","r",encoding="utf-8") as f:
    text=f.read()

start_text=text.find("Chapter 1")
end_text=text.rfind(" End of the Project Gutenberg")
text=text[start_text:end_text]

print("Length of dataset in characters:",len(text))
print(text[:1000])


#Builf Vocabulary 
chars=sorted(list(set(text)))
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}

encode=lambda s: [stoi[c] for c in s]
decode= lambda l: ''.join([itos[i] for i in l])

class TextDataset(Dataset):
    def __init__(self,data):
        self.data=data 

    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self,idx):
        chunk=self.data[idx]
        x=chunk[:-1] 
        y=chunk[1:]
        return x,y 
    
