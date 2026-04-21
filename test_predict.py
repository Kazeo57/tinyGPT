import json 
import torch 
from model import TinyBert , TinyGPT
#model_path="checkpoint_30.pth"
model_path="checkpoint_50_gpt.pth"

#with open('data/vocab.json','r',encoding='utf-8') as f:
with open('words_vocab.json','r',encoding='utf-8') as f:
    vocab=json.load(f)
vocab_size=len(vocab)
print("Vocab size",vocab_size)
block_size=64

#print(vocab)
#encode=lambda s: [vocab[c] for c in s]
encode=lambda s: [vocab[w] for w in s.split()]
#decode= lambda l: ''.join([itos[i] for i in l])
##decode=lambda l: "".join([k for id in l for k,v in vocab.items() if id==v])
itos={v:k for k,v in vocab.items()}
decode=lambda l: " ".join(itos[i] for i in l)

#model=TinyBert(vocab_size=vocab_size,block_size=block_size,n_emb=128)

model=TinyGPT(embed_dim=128,vocab_size=vocab_size,num_heads=4,num_layers=8)
state_dict=torch.load(model_path)
#print("State Dict",state_dict)
#state_dict={k.replace("module.",""):v for k,v in state_dict.items()}
model.load_state_dict(state_dict['model_state_dict'])
#model.load_state_dict(torch.load(model_path))

#print("Model",model)
#model.eval()
def generate(model,start_seq="All this was acknowledged".lower(),length=100):
    model.eval()
    with torch.no_grad():  
        current_id=encode(start_seq) 
        for i in range(length):
            x=torch.tensor(current_id[-block_size:], dtype=torch.long)
            #x=x.to()
            logits=model(x)
            #print("LOGITS", logits)
            ##probs=torch.nn.functional.softmax(logits[:,-1],dim=-1)
            ##next_id=torch.multinomial(probs,num_samples=1)
            ###
            last_logits = logits.view(-1, logits.size(-1))[-1] 

            # 2. Top-K (renvoie deux vecteurs 1D de taille 50)
            v, i = torch.topk(last_logits, 50)

            # 3. Softmax pour transformer les scores en probabilités
            probs = torch.nn.functional.softmax(v, dim=-1)

            # 4. On pioche un index LOCAL (entre 0 et 49)
            idx_in_topk = torch.multinomial(probs, num_samples=1).item()

            # 5. On récupère le VRAI ID (celui du vocabulaire global)
            # Comme 'i' est en 1D, on utilise un seul index
            next_id = i[idx_in_topk].item()
            ###
            #print("Next_id",next_id.tolist())
            #next_char=decode(next_id.tolist()[0])
            #print(f"Char {i}",next_char)
            #start_seq+= next_char 
            #print(f"seq {i}",start_seq)
            current_id.append(next_id)
    #return start_seq
    return decode(current_id)

print(generate(model))
print(""" Reference: \n
      All this was acknowledged to Mrs. Gardiner; and, after relating the
circumstances, she thus went on:--“I am now convinced, my dear aunt,
that I have never been much in love; for had I really experienced that
pure and elevating passion, I should at present detest his very name,
and wish him all manner of evil.""")