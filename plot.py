import matplotlib.pyplot as plt 
import pandas as pd 
import re 

data=pd.read_csv('Training_metadata.csv')
#data=data['Loss'].map(lambda x: {"track_loss":x.item()})
data["Loss"] = data["Loss"].apply(
    lambda x: float(re.search(r"tensor\(([^,]+)", x).group(1))
)

losses = data["Loss"]
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss view")

plt.savefig("training_loss.png", dpi=300, bbox_inches="tight")