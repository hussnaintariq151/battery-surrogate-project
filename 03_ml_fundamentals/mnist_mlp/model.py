"""
model.py
--------
Yahan hum apna MLP (Multi-Layer Perceptron) define karte hain.

Architecture:
    Input  : 784  (28x28 pixels flat)
    Hidden1: 512  neurons  + ReLU + Dropout
    Hidden2: 256  neurons  + ReLU + Dropout
    Output : 10   neurons  (digit 0-9 ka score)
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# STEP 1: nn.Module kyun inherit karte hain?
# ─────────────────────────────────────────────
# PyTorch mein har model nn.Module ka bacha hota hai.
# Isse yeh fayde milte hain:
#   - model.parameters() → saare weights automatically milte hain
#   - model.train() / model.eval() → mode switch hota hai
#   - Saving/loading asaan ho jaata hai
#   - Autograd automatically kaam karta hai

class MLP(nn.Module):

    # ─────────────────────────────────────────
    # __init__: layers define karo
    # ─────────────────────────────────────────
    def __init__(self, input_size=784, hidden1=512, hidden2=256, num_classes=10, dropout_rate=0.2):
        """
        Args:
            input_size  : 784 (28x28 pixels)
            hidden1     : pehli hidden layer ke neurons
            hidden2     : doosri hidden layer ke neurons
            num_classes : 10 (digits 0-9)
            dropout_rate: training mein randomly neurons band karo
                          (overfitting rokne ke liye)
        """
        # Pehle parent class ka __init__ chalao — ZAROORI hai
        super(MLP, self).__init__()

        # ── Layer 1: Input → Hidden1 ──────────────────────
        # nn.Linear(in, out) → ek fully connected layer
        # Andar kya hota hai: output = input @ W.T + b
        #   W (weight matrix): shape [512, 784]
        #   b (bias vector)  : shape [512]
        # Yeh W aur b hi "seekhne" wale parameters hain
        self.fc1 = nn.Linear(input_size, hidden1)

        # ── Activation: ReLU ─────────────────────────────
        # ReLU(x) = max(0, x)
        # Negative values zero ho jaati hain
        # Positive values waise hi rehti hain
        # Kyun? Non-linearity ke bina MLP sirf linear equation hai
        self.relu1 = nn.ReLU()

        # ── Dropout ──────────────────────────────────────
        # Training mein 20% neurons randomly "off" ho jaate hain
        # Isse model ek hi path par depend nahi karta
        # → Overfitting kam hoti hai
        # NOTE: Eval mode mein dropout automatically band ho jaata hai
        self.dropout1 = nn.Dropout(p=dropout_rate)

        # ── Layer 2: Hidden1 → Hidden2 ───────────────────
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)

        # ── Layer 3: Hidden2 → Output ─────────────────────
        # 10 outputs → har digit (0-9) ka ek score (logit)
        # Yahan koi activation nahi → kyunke loss function
        # (CrossEntropyLoss) andar se Softmax lagata hai
        self.fc3 = nn.Linear(hidden2, num_classes)

    # ─────────────────────────────────────────
    # forward: data ka flow define karo
    # ─────────────────────────────────────────
    # Yeh method automatically call hoti hai jab tum
    # model(x) likhte ho
    #
    # Autograd ka kaam:
    #   - Forward pass mein PyTorch har operation track karta hai
    #   - Baad mein loss.backward() call karo
    #   - Autograd khud gradients calculate kar leta hai
    #   - Tumhe kuch manually nahi karna!

    def forward(self, x):
        """
        Args:
            x: input tensor, shape [batch_size, 1, 28, 28]

        Returns:
            logits: shape [batch_size, 10]
                    (raw scores, softmax nahi lagi abhi)
        """

        # ── Flatten ───────────────────────────────────────
        # Image abhi [batch, 1, 28, 28] hai
        # Linear layer ke liye chahiye: [batch, 784]
        # x.view(batch_size, -1) → -1 matlab "baaki sab ek mein"
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)       # [batch, 784]

        # ── Layer 1 ───────────────────────────────────────
        x = self.fc1(x)                   # [batch, 512]
        x = self.relu1(x)                 # negatives → 0
        x = self.dropout1(x)              # 20% neurons off (training mein)

        # ── Layer 2 ───────────────────────────────────────
        x = self.fc2(x)                   # [batch, 256]
        x = self.relu2(x)
        x = self.dropout2(x)

        # ── Output Layer ──────────────────────────────────
        x = self.fc3(x)                   # [batch, 10]

        # Yahan softmax nahi lagayi
        # CrossEntropyLoss automatically karta hai yeh kaam
        return x                          # logits return karo


# ─────────────────────────────────────────────
# STEP 2: Helper function — model ka summary
# ─────────────────────────────────────────────
def count_parameters(model):
    """
    Model mein total trainable parameters count karo.
    Har Linear layer mein: (in * out) weights + out biases
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


# ─────────────────────────────────────────────
# Sanity Check — seedha run karo
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=== Model Check ===\n")

    # Model banao
    model = MLP()
    print(model)
    print()

    # Parameters count karo
    params = count_parameters(model)
    print(f"Total trainable parameters: {params:,}")
    # Expected: ~567,050

    # Ek fake batch se test karo
    # requires_grad=False → sirf forward pass, backward nahi
    dummy_input = torch.randn(32, 1, 28, 28)   # 32 fake images
    output = model(dummy_input)

    print(f"\nInput shape  : {dummy_input.shape}")   # [32, 1, 28, 28]
    print(f"Output shape : {output.shape}")          # [32, 10]
    print(f"\nModel ready hai!")

    # Autograd check
    print("\n=== Autograd Check ===")
    for name, param in model.named_parameters():
        print(f"{name:20s} | requires_grad: {param.requires_grad} | shape: {list(param.shape)}")