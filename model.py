import torch 
import torch.nn as nn 

class TinyGPT(nn.Module):
    def __init__(self,vocab_size,block_size,n_emb):
        super(TinyGPT,self).__init__()
        self.embedding=nn.Embedding(vocab_size,n_emb)
        self.encoder=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=n_emb,nhead=n_emb//64,dim_feedforward=n_emb*4),
            num_layers=12)
        self.norm=nn.LayerNorm(n_emb)
        self.head=nn.Linear(n_emb,vocab_size,bias=False)

    def forward(self,idx):
        tokens=self.embedding(idx)
        tokens=self.encoder(tokens)
        tokens=self.norm(tokens)
        return self.head(tokens)