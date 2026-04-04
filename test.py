import json
with open('data/vocab.json','r',encoding='utf-8') as f:
    vocab=json.load(f)
decode=lambda l: "".join([k for id in l for k,v in vocab.items() if id==v])
print(decode([2,6,9]))