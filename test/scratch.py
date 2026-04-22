import torch
import torch.nn as nn 
vocab_size=23
dim=3
batch_size=2 #1
#inputs=torch.tensor([2,3])
inputs=torch.tensor([[2,3],[4,8]])
#inputs=inputs.reshape(1,-1)
#assert batch_size>= int(inputs.shape[0])
#inputs=inputs.view(batch_size,inputs.size(0),inputs.size(1))
print("Input shape",inputs.size())
embedding= nn.Embedding(num_embeddings=vocab_size,embedding_dim=dim)
embedded_input=embedding(inputs)
print("Ebedded Shape",embedded_input.shape)

Wq=nn.Linear(in_features=dim,out_features=dim)
Wk=nn.Linear(in_features=dim,out_features=dim)
Wv=nn.Linear(in_features=dim,out_features=dim)
#q=Wq(embedded_input)
#print("Weights q: ",q.shape)
mask=torch.tril(torch.ones_like(embedded_input),diagonal=0).bool()
mask=torch.masked_fill(embedded_input,mask==0,float('-inf'))
print("Mask :",mask)
print("Mask shape: ",mask.shape)
class SelfAttention(nn.Module):
    def __init__(self,dim_model):
        #super().__init__(SelfAttention,self)
        self.Wq=nn.Linear(in_features=dim_model,out_features=dim_model)
        self.Wk=nn.Linear(in_features=dim_model,out_features=dim_model)
        self.Wv=nn.Linear(in_features=dim_model,out_features=dim_model)

    def forward(self,X):
        q=self.Wq(X)
        k=self.Wk(X)
        v=self.Wv(X)

        first=torch.matmul(q,torch.transpose(k,-2,-1))
        #first=first/torch.sqrt(q.size(-1))
        first=first/(q.size(-1))**0.5
        first=torch.softmax(first,dim=-1)
        score=torch.matmul(first,v)
        return score 
    
    
    #Wq=nn.Linear(in_features=dim,out_features=dim)
    #Wk=nn.Linear(in_features=dim,out_features=dim)
    #Wv=nn.Linear(in_features=dim,out_features=dim)

"""
    q=Wq(embedded_input)
    k=Wk(embedded_input)
    v=Wv(embedded_input)

    first=torch.matmul(q,torch.transpose(k,-2,-1))
    #first=first/torch.sqrt(q.size(-1),dtype=first.dtype)
    first=first/(q.size(-1))**0.5
    first=torch.softmax(first,dim=-1)
    score=torch.matmul(first,v)
    print("Score :",score)
    print("Score shape",score.shape)
"""

class MaskedAttention:
    def __init__(self,embed_dim):
        #super().__init__(MaskedAttention,self)
        self.Wq=nn.Linear(embed_dim,embed_dim)
        self.Wk=nn.Linear(embed_dim,embed_dim)
        self.Wv=nn.Linear(embed_dim,embed_dim)


    def forward(self,X):
        q=self.Wq(X)
        k=self.Wk(X)
        v=self.Wv(X)

        X=torch.matmul(q,k.transpose(-2,-1))/torch.sqrt(q.size(-1))
        mask=torch.tril(torch.ones_like(X),diagonal=0)#.bool()
        mask=torch.masked_fill(X,mask==0,float('-inf'))
        #X=(X+mask)/(q.size(-1))**0.5
        weights=torch.softmax(mask,dim=-1)
        score=torch.matmul(weights,v)
        return score
        

class MultiHeadSelfAttention:
    def __init__(self,embed_dim,num_heads):
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.head_dim=embed_dim//num_heads
        self.Wq=nn.Linear(embed_dim,embed_dim)
        self.Wk=nn.Linear(embed_dim,embed_dim)
        self.Wv=nn.Linear(embed_dim,embed_dim)
        self.out_proj=nn.Linear(embed_dim,embed_dim)

    def forward(self,X):
        q=self.Wq(X)
        q=q.view(q.size(0),q.size(1),self.num_heads,self.head_dim).transpose(1,2)
        k=self.Wk(X)
        k=k.view(k.size(0),k.size(1),self.num_heads,self.head_dim).transpose(1,2)
        v=self.Wv(X)
        v=v.view(v.size(0),v.size(1),self.num_heads,self.head_dim).transpose(1,2)

        score=q.matmul(k.transpose(-2,-1))
        score=score/torch.sqrt(q.size(-1))
        weights=torch.softmax(score,dim=-1)
        output=torch.matmul(weights,v)
        output.transpose(1,2)
        output=output.view(output.size(0),output.size(1),self.embed_dim)
        return self.out_proj(output)
    

class MaskedMultiHeadAttention:
    def __init__(self,embed_dim,num_heads):
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.head_dim=self.embed_dim//num_heads
        self.Wq=nn.Linear(embed_dim,embed_dim)
        self.Wk=nn.Linear(embed_dim,embed_dim)
        self.Wv=nn.Linear(embed_dim,embed_dim)
        self.out_proj=nn.Linear(embed_dim,embed_dim)
    def forward(self,X):
        q=self.Wq(X)
        k=self.Wk(X)
        v=self.Wv(X)

        q=q.view(q.size(0),q.size(1),self.num_heads,self.head_dim).transpose(1,2)
        k=k.view(k.size(0),k.size(1),self.num_heads,self.head_dim).transpose(1,2)
        v=v.view(v.size(0),v.size(1),self.num_heads,self.head_dim).transpose(1,2)

        scores=q.matmul(k.transpose(-2,-1))/torch.sqrt(q.size(-1))
        mask=torch.tril(torch.ones_like(scores),diagonal=0).bool()
        mask=torch.masked_fill(mask==0,float('-inf'))
        weights=torch.softmax(scores,dim=-1)
        score=torch.matmul(weights,v)
        score=score.transpose(1,2)
        score=score.view(score.size(0),score.size(1),self.embed_dim)
        return self.out_proj(score)
    

class DecoderBlock:
    def __init__(self,embed_dim,num_heads,golden_ratio=4):
        self.mmha=MaskedMultiHeadAttention(
            embed_dim,
            num_heads,
        )
        self.ffn=nn.Sequential(
            nn.Linear(embed_dim,embed_dim*golden_ratio),
            nn.GELU(),
            nn.Linear(golden_ratio*embed_dim,embed_dim),
            nn.Dropout(0.1)
        )
        self.layer_norm1=nn.LayerNorm(embed_dim)
        self.layer_norm2=nn.LayerNorm(embed_dim)
        self.out_layer=nn.Linear(embed_dim,embed_dim)
    def forward(self,X):
        #PreLayerNorm
        norm1=self.layer_norm1(X)
        attn_out=X+self.mmha(norm1)
        norm2=self.layer_norm2(attn_out)
        ffn_out=attn_out+ self.ffn(norm2)
        return ffn_out
    
def positional_encoding(embedding):
    pos=torch.arange(embedding.size(1),device=embedding.device)
    pos_embedding=embedding.clone().detach()
    indices=torch.arange(pos_embedding.size(-1),device=pos_embedding.device)
    mask=indices%2==0
    pos_embedding[...,mask]+=torch.math.sin(pos)/(10000**(2*indices[mask]/pos_embedding.size(-1)))
    return embedding
    
class Decoder:
    def __init__(self,vocab_size,embed_dim,num_heads,num_layers,decoder_block):
        self.embedding=nn.Embedding(vocab_size,embed_dim)
        self.decoder_stack=nn.ModuleList([DecoderBlock(embed_dim,num_heads) for _ in range(num_layers)])
        self.layer_norm=nn.LayerNorm(embed_dim)
        self.final_out_proj=nn.Linear(embed_dim,embed_dim)

    def forward(self,X):
        X=self.embedding(X)
        X=positional_encoding(X)
        X=self.decoder_stack(X)
        X=self.layer_norm(X)
        return self.final_out_proj(X)
    
#torch.index_select()
#mask= mask[embedded_input%2==0]
positional_embedding=embedded_input.clone().detach()
indices=torch.arange(positional_embedding.size(-1),device=positional_embedding.device)
mask=indices%2==0
pos=2
positional_embedding[...,mask]+= torch.math.sin(pos)/(10000**(2*indices[mask]/dim))
positional_embedding[...,~mask]+= torch.math.cos(pos)/10000**(2*indices[~mask]/dim)
print("Pos 2 Shape: ",positional_embedding.shape)
print("Pos embedding : ",positional_embedding)
        

    

    





        



    



