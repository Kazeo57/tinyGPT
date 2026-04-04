#dataLoader 
from torch.utils.data import Dataset 
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
    