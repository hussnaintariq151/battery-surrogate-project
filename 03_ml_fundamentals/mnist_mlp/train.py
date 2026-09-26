"""
train.py
--------
Yahan hum MLP ko MNIST pe train karte hain.

Is file mein yeh sab cover hoga:
  - Training loop (har epoch, har batch)
  - Loss tracking
  - Validation (test accuracy)
  - Checkpointing (best model save karna)
  - Autograd ka practical use
"""

import torch
import torch.nn as nn
import os

from dataset import get_dataloaders
from model   import MLP, count_parameters


# ═════════════════════════════════════════════
# 1. DEVICE SETUP
# ═════════════════════════════════════════════
# GPU available hai to GPU use karo, warna CPU
# .to(device) se model aur data GPU/CPU pe jaate hain
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ═════════════════════════════════════════════
# 2. HYPERPARAMETERS
# ═════════════════════════════════════════════
# Yeh values tum control karte ho — model seekhne ke rules
BATCH_SIZE    = 64      # ek baar mein kitni images
LEARNING_RATE = 0.001   # har step mein kitna update karo
EPOCHS        = 10      # poora data kitni baar dekhna hai
CHECKPOINT_DIR = './checkpoints'  # model save karne ki jagah


# ═════════════════════════════════════════════
# 3. TRAIN — EK EPOCH
# ═════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    """
    Ek poori epoch train karta hai.

    Args:
        model     : hamara MLP
        loader    : training DataLoader
        criterion : loss function (CrossEntropyLoss)
        optimizer : Adam optimizer
        epoch     : current epoch number (sirf print ke liye)

    Returns:
        avg_loss : is epoch ki average loss
        accuracy : is epoch ki training accuracy
    """

    # ── model.train() kyun? ───────────────────────────────
    # Dropout aur BatchNorm training mode mein alag behave karte hain
    # model.train() → Dropout ON (neurons randomly off hote hain)
    # model.eval()  → Dropout OFF (sab neurons active)
    model.train()

    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    # ── Har batch pe iterate karo ────────────────────────
    for batch_idx, (images, labels) in enumerate(loader):

        # Data ko device pe bhejo (GPU ya CPU)
        images = images.to(device)   # [64, 1, 28, 28]
        labels = labels.to(device)   # [64]

        # ── STEP 1: Purane gradients saaf karo ──────────
        # Kyun? PyTorch gradients accumulate karta hai by default
        # Agar zero nahi kiya to purane + naye gradients add ho jaayenge
        # → Weights galat direction mein jayenge
        optimizer.zero_grad()

        # ── STEP 2: Forward pass ─────────────────────────
        # model(images) → forward() method call hoti hai
        # Autograd yahan se computation graph banana shuru karta hai
        # Har operation track hoti hai (requires_grad=True wali tensors)
        outputs = model(images)       # [64, 10] logits

        # ── STEP 3: Loss calculate karo ──────────────────
        # CrossEntropyLoss kya karta hai?
        #   1. Softmax lagata hai outputs pe
        #   2. Log probability nikalti hai
        #   3. Sahi class ki probability maximize karta hai
        # outputs: [64, 10] raw scores
        # labels : [64]    actual digit (0-9)
        loss = criterion(outputs, labels)

        # ── STEP 4: Backward pass (Autograd) ─────────────
        # loss.backward() → chain rule se har parameter ka
        # gradient calculate hota hai
        # param.grad mein store ho jaata hai
        # Tumhe kuch manually nahi karna!
        loss.backward()

        # ── STEP 5: Weights update karo ──────────────────
        # Adam optimizer: param = param - lr * param.grad
        # (thoda zyada smart — momentum bhi use karta hai)
        optimizer.step()

        # ── Stats track karo ─────────────────────────────
        total_loss += loss.item()     # .item() → Python float mein badlo
                                      # .item() kyun? tensor se gradient
                                      # detach hoti hai — memory save hoti hai

        # Predictions nikalo
        # torch.argmax → sabse bada score wala index = predicted digit
        _, predicted = torch.max(outputs, dim=1)   # [64]
        correct       += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Har 200 batch pe progress print karo
        if (batch_idx + 1) % 200 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} "
                  f"| Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total_samples
    return avg_loss, accuracy


# ═════════════════════════════════════════════
# 4. EVALUATE — TEST SET PE CHECK KARO
# ═════════════════════════════════════════════
def evaluate(model, loader, criterion):
    """
    Test set pe model ka performance check karo.

    Returns:
        avg_loss : test loss
        accuracy : test accuracy (%)
    """

    # ── model.eval() kyun? ───────────────────────────────
    # Dropout band ho jaata hai → sab neurons active
    # BatchNorm bhi alag behave karta hai eval mein
    model.eval()

    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    # ── torch.no_grad() kyun? ────────────────────────────
    # Evaluation mein hume gradients nahi chahiye
    # no_grad() → computation graph nahi banta
    # → Memory aur speed dono better hoti hain
    # → Yeh hai "detach" ka practical use case
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs  = model(images)
            loss     = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted  = torch.max(outputs, dim=1)
            correct       += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total_samples
    return avg_loss, accuracy


# ═════════════════════════════════════════════
# 5. CHECKPOINT — BEST MODEL SAVE KARO
# ═════════════════════════════════════════════
def save_checkpoint(model, optimizer, epoch, accuracy, filepath):
    """
    Model ka state save karo taake baad mein resume kar sako.

    Checkpoint mein kya save hota hai:
        - model weights (state_dict)
        - optimizer state (momentum values)
        - current epoch
        - best accuracy
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch'     : epoch,
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy'  : accuracy,
    }, filepath)
    print(f"  ✓ Checkpoint saved: {filepath} (accuracy: {accuracy:.2f}%)")


# ═════════════════════════════════════════════
# 6. MAIN TRAINING LOOP
# ═════════════════════════════════════════════
def main():
    print("=" * 55)
    print("       MNIST MLP Training")
    print("=" * 55)

    # ── Data load karo ───────────────────────────────────
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    print(f"Train batches : {len(train_loader)}")
    print(f"Test batches  : {len(test_loader)}\n")

    # ── Model banao aur device pe bhejo ──────────────────
    model = MLP().to(device)
    print(f"Parameters    : {count_parameters(model):,}\n")

    # ── Loss Function ────────────────────────────────────
    # CrossEntropyLoss = Softmax + NegativeLogLikelihood
    # Multi-class classification ke liye standard choice
    criterion = nn.CrossEntropyLoss()

    # ── Optimizer ────────────────────────────────────────
    # Adam: Adaptive Moment Estimation
    # Simple SGD se better — learning rate adjust karta hai
    # model.parameters() → saare weights automatically milte hain
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ── History track karo (plotting ke liye) ────────────
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss' : [], 'test_acc' : []
    }

    best_accuracy = 0.0

    # ── Epochs ───────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 40)

        # Train karo
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch
        )

        # Evaluate karo
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        # History save karo
        history['train_loss'].append(train_loss)
        history['train_acc' ].append(train_acc)
        history['test_loss' ].append(test_loss)
        history['test_acc'  ].append(test_acc)

        # Results print karo
        print(f"\n  Train → Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"  Test  → Loss: {test_loss:.4f}  | Acc: {test_acc:.2f}%")

        # ── Checkpointing ────────────────────────────────
        # Sirf best model save karo
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            save_checkpoint(
                model, optimizer, epoch, test_acc,
                filepath=f"{CHECKPOINT_DIR}/best_model.pth"
            )

    # ── Final Summary ────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  Training Complete!")
    print(f"  Best Test Accuracy : {best_accuracy:.2f}%")
    print(f"  Target             : 90.00%")
    print(f"  Status             : {'✓ PASSED' if best_accuracy >= 90 else '✗ FAILED'}")
    print("=" * 55)

    return history


# ─────────────────────────────────────────────
# Run karo
# ─────────────────────────────────────────────
if __name__ == '__main__':
    main()