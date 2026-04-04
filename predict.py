import json 
import torch 


with open('data/vocab.json','r',encoding='utf-8') as f:
    vocab=json.load(f)
#print(vocab)
encode=lambda s: [vocab[c] for c in s]
#decode= lambda l: ''.join([itos[i] for i in l])
decode=lambda l: "".join([k for id in l for k,v in vocab.items() if id==v])

model=torch.load("checkpoint_9.ckpt",map_location=None)
def generate(model,start_seq="It is a truth universally acknowledged",length=100):
    model.eval()
    with torch.no_grad():
        for _ in range(length):
            x=torch.tensor([encode(start_seq)], dtype=torch.long)
            logits=model(x)
            probs=torch.nn.functional.softmax(logits[:,-1],dim=-1)
            next_id=torch.mulnomial(probs,num_samples=1)
            next_char=decode(next_id.tolist())
            start_seq+= next_char 
    return start_seq

print(generate(model))