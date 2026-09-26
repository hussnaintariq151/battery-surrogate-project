# MNIST MLP — From Scratch

Ek Multi-Layer Perceptron jo MNIST handwritten digits ko 98%+ accuracy se pehchanta hai.  
Koi high-level wrapper nahi — har cheez khud likhi hai: Dataset, Model, Training Loop.

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **98.14%** |
| Train Accuracy | 98.74% |
| Target | 90.00% |
| Epochs | 10 |
| Parameters | 535,818 |

---

## Architecture

```
Input (784)  →  Hidden1 (512)  →  Hidden2 (256)  →  Output (10)
              ReLU + Dropout     ReLU + Dropout     CrossEntropyLoss
```

MNIST image `28×28 = 784 pixels` hai. Inhe flat vector banake network mein daalte hain.  
Har hidden layer ke baad ReLU non-linearity add karta hai aur Dropout overfitting rokta hai.

---

## Project Structure

```
mnist_mlp/
├── dataset.py   # Custom Dataset + DataLoader
├── model.py     # Custom nn.Module (MLP)
├── train.py     # Training loop, checkpointing
├── checkpoints/
│   └── best_model.pth   # Best saved model
└── data/                # MNIST (auto-downloaded)
```

---

## Concepts Covered

**Custom Dataset & DataLoader**
- `__init__`, `__len__`, `__getitem__` khud likhe
- `batch_size=64`, `shuffle=True` training mein

**Custom nn.Module**
- `super().__init__()` — parent class initialize karna
- `__init__` mein layers define, `forward()` mein data flow
- ReLU activation — non-linearity ke liye
- Dropout (p=0.2) — overfitting rokne ke liye

**Training Loop**
```python
optimizer.zero_grad()   # 1. Purane gradients saaf karo
output = model(x)       # 2. Forward pass
loss = criterion(...)   # 3. Loss calculate karo
loss.backward()         # 4. Autograd — gradients nikalo
optimizer.step()        # 5. Weights update karo
```

**Autograd**
- `loss.backward()` — chain rule automatically apply hoti hai
- `torch.no_grad()` — evaluation mein graph nahi banta, memory bachti hai
- `.item()` — tensor se gradient detach karke Python float milta hai

**Checkpointing**
- Har epoch ke baad test accuracy check
- Sirf best model save hota hai `checkpoints/best_model.pth` mein
- Save hota hai: weights, optimizer state, epoch, accuracy

---

## How to Run

```bash
# 1. Dependencies install karo
pip install torch torchvision

# 2. Dataset check karo
python dataset.py

# 3. Training shuru karo
python train.py
```

---

## Training Progress

```
Epoch  1  → Test Acc: ~95%   (model seekhna shuru)
Epoch  5  → Test Acc: ~97%   (tezi se improve)
Epoch 10  → Test Acc: 98.14% (converge)
```

Loss har epoch mein giri, train aur test accuracy close rahin —  
model ne generalize kiya, sirf training data yaad nahi kiya.

---
