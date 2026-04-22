# tinyGPT

## Overview 
This work aim to study GPT architecture and make an optimization roadmap for Training and Model size constraint.

## Full Basic Training:
For a basic full training, make this:
### Get Data & Build Tokenizer:
```bash
python3 data/get_data.py
```

### Train the model:
```bash
python3 main.py 
```

### Make inference with that:
```bash
pyhon3 predict.py
```

### Experimentation Results:
**Loss:** 0.5604
**Epochs:** 50
![Training loss](training_loss2.png)

# Technologies
# Technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Torchview](https://img.shields.io/badge/Torchview-Library-orange?style=for-the-badge)

* **Core:** Python 3.9+
* **Framework:** PyTorch (Neural Network implementation)
* **Visualization:** Torchview (Architecture graph rendering)


