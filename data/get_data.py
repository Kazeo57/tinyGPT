import torch
import subprocess
import json



#subprocess.run(["wget", "https://www.gutenberg.org/files/1342/1342-0.txt" ,"-O" ,"input.txt"])
subprocess.run(["curl","-L","-o","input.txt","https://www.gutenberg.org/files/1342/1342-0.txt"])


with open("input.txt","r",encoding="utf-8") as f:
    text=f.read()

print(len(text))

#start_text=text.find("Chapter 1")
#end_text=text.rfind("End of the Project Gutenberg")
#text=text[start_text:end_text]
#print("Text", text)

print("Length of dataset in characters:",len(text))
#print(text[:1000])


#Build Vocabulary 
chars=sorted(list(set(text)))
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}

with open("vocab.json","w") as f:
    json.dump(stoi,f,indent=4)





