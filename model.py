import torch 
import torch.nn as nn 

class TinyBert(nn.Module):
    def __init__(self,vocab_size,block_size,n_emb):
        super(TinyBert,self).__init__()
        self.embedding=nn.Embedding(vocab_size,n_emb)
        self.encoder=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=n_emb,nhead=n_emb//64,dim_feedforward=n_emb*4),
            num_layers=12)
        #self.decoder=nn.TransformerDecoder(
            #nn.TransformerDecoderLayer(d_model=n_emb,nhead=n_emb//64,dim_feedforward=n_emb*4),
            #num_layers=12)
        self.norm=nn.LayerNorm(n_emb)
        self.head=nn.Linear(n_emb,vocab_size,bias=False)

    def forward(self,idx):
        tokens=self.embedding(idx)
        tokens=self.encoder(tokens)
        #tokens=self.decoder(tokens)
        tokens=self.norm(tokens)
        return self.head(tokens)

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super(MaskedMultiHeadAttention,self).__init__()
        self.embed_dim=embed_dim
        self.Wq=nn.Linear(embed_dim,embed_dim)
        self.Wk=nn.Linear(embed_dim,embed_dim)
        self.Wv=nn.Linear(embed_dim,embed_dim)
        self.num_heads=num_heads
        self.head_dim=embed_dim//num_heads
        self.decode_out_proj=nn.Linear(embed_dim,embed_dim)


    def forward(self,X):
        if X.dim()==2:
            X=X.unsqueeze(0)
            squeeze=True 
        else:
            squeeze=False
        q=self.Wq(X)
        q=q.view(q.size(0),q.size(1),self.num_heads,self.head_dim).transpose(1,2)
        k=self.Wk(X)
        k=k.view(k.size(0),k.size(1),self.num_heads,self.head_dim).transpose(1,2)
        v=self.Wv(X)
        v=v.view(v.size(0),v.size(1),self.num_heads,self.head_dim).transpose(1,2)
        score=q.matmul(k.transpose(-2,-1))
        score=score/torch.math.sqrt(k.size(-1))
        
        mask=torch.tril(torch.ones_like(score),diagonal=0).bool()
        
        mask=torch.masked_fill(score,mask==0, float('-inf'))
        weights=torch.softmax(mask,dim=-1)
        score=weights.matmul(v)
        score=score.transpose(1,2)
        #score=score.contiguous().view(score.size(0),score.size(1),self.embed_dim)
        score=score.reshape(score.size(0),score.size(1),self.embed_dim)
        out=self.decode_out_proj(score)
        if squeeze:
            out=out.squeeze(0)

        return out
        




class DecoderBlock(nn.Module):
    def __init__(self,embed_dim,num_heads,golden_ratio=4):
        super(DecoderBlock,self).__init__()
        self.mmha=MaskedMultiHeadAttention(embed_dim,num_heads)
        self.ffn=nn.Sequential(
            nn.Linear(embed_dim,golden_ratio*embed_dim),
            nn.GELU(),
            nn.Linear(golden_ratio*embed_dim,embed_dim),
            nn.Dropout(0.1)
        )
        self.layer_norm1=nn.LayerNorm(embed_dim)
        self.layer_norm2=nn.LayerNorm(embed_dim)
        self.layer_norm=nn.LayerNorm(embed_dim)
    
    def forward(self,X):
        norm1=self.layer_norm1(X)
        attn_out=self.mmha(norm1)
        attn_out+=X
        ffn_out=self.ffn(self.layer_norm2(attn_out))
        ffn_out+=attn_out
        return self.layer_norm(ffn_out)

def positional_encoding(X):
    pos=torch.arange(X.size(-2),device=X.device)
    pos=pos.reshape(-1,1)
    #pos=pos.view(pos.size(-2),pos.size(-1))
    pos_embedding=X.clone().detach()
    indices=torch.arange(pos_embedding.size(-1),device=X.device)
    mask=indices%2==0
    #pos=torch.zeros(X.size(1),X.size(2))
    pos_embedding[...,mask]+= torch.sin(pos/10000**(2*indices[mask]/pos_embedding.size(-1)))#.unsqueeze(0)
    pos_embedding[...,~mask]+= torch.cos(pos/10000**(2*indices[~mask]/pos_embedding.size(-1)))#.unsqueeze(0)
    return pos_embedding 
    
class TinyGPT(nn.Module):
    def __init__(self,embed_dim,vocab_size,num_heads,num_layers,golden_ratio=4):
        super(TinyGPT,self).__init__()
        self.embedding=nn.Embedding(vocab_size,embed_dim)
        #self.pos_embedding=positional_encoding()
        self.decoder_stack=nn.Sequential(*[DecoderBlock(embed_dim,num_heads,golden_ratio=4) for _ in range(num_layers)])
        self.final_layer_norm=nn.LayerNorm(embed_dim)
        self.final_out_proj=nn.Linear(embed_dim,vocab_size)

    def forward(self,X):
        embedding=self.embedding(X)
        embedding=positional_encoding(embedding)
        causal_embedding=self.decoder_stack(embedding)
        causal_embedding=self.final_layer_norm(causal_embedding)
        return self.final_out_proj(causal_embedding)
    
