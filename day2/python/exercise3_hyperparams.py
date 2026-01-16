"""
Exercise 3: Learning Rate and Epochs
=====================================
Experiment with different learning rates and number of epochs

Run: python exercise3_hyperparams.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ============================================
# Step 1: Create the training data
# ============================================
x_data = torch.linspace(-2, 2, 101).reshape(-1, 1)
y_data = 2 * x_data - x_data ** 3

print("Data: 101 points, function y = 2x - x^3\n")


# ============================================
# Step 2: Define the neural network
# ============================================
class SimpleNetwork(nn.Module):
    """4-layer network with Xavier initialization"""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 20)
        self.layer3 = nn.Linear(20, 10)
        self.layer4 = nn.Linear(10, 1)

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        x = self.layer4(x)
        return x


# ============================================
# Step 3: Training function
# ============================================
def train(model, lr, epochs, use_full_batch=True):
    """
    Train model with given learning rate and epochs

    use_full_batch=True: Process all data at once (smoother)
    use_full_batch=False: Process one sample at a time (noisier)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []

    for epoch in range(epochs):
        if use_full_batch:
            # Full batch: use all data at once
            prediction = model(x_data)
            loss = loss_fn(prediction, y_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        else:
            # Mini-batch (size 1): process one sample at a time
            epoch_loss = 0
            for i in range(len(x_data)):
                x_sample = x_data[i:i+1]
                y_sample = y_data[i:i+1]

                pred = model(x_sample)
                loss = loss_fn(pred, y_sample)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            losses.append(epoch_loss / len(x_data))

    return losses


# ============================================
# Step 4: Experiment with different learning rates
# ============================================
print("="*50)
print("EXPERIMENT 1: Different Learning Rates")
print("="*50)

learning_rates = [0.1, 0.01, 0.001, 0.0001]
results_lr = {}

for lr in learning_rates:
    torch.manual_seed(42)
    model = SimpleNetwork()
    losses = train(model, lr=lr, epochs=500)
    results_lr[lr] = losses
    print(f"lr={lr}: Final loss = {losses[-1]:.6f}")


# ============================================
# Step 5: Experiment with different epochs
# ============================================
print("\n" + "="*50)
print("EXPERIMENT 2: Different Number of Epochs")
print("="*50)

epoch_values = [100, 500, 1000, 2000]
results_epochs = {}

for epochs in epoch_values:
    torch.manual_seed(42)
    model = SimpleNetwork()
    losses = train(model, lr=0.001, epochs=epochs)
    results_epochs[epochs] = losses
    print(f"epochs={epochs}: Final loss = {losses[-1]:.6f}")


# ============================================
# Step 6: Compare full batch vs mini-batch
# ============================================
print("\n" + "="*50)
print("EXPERIMENT 3: Full Batch vs Mini-batch")
print("="*50)

torch.manual_seed(42)
model_full = SimpleNetwork()
losses_full = train(model_full, lr=0.001, epochs=300, use_full_batch=True)
print(f"Full batch: Final loss = {losses_full[-1]:.6f}")

torch.manual_seed(42)
model_mini = SimpleNetwork()
losses_mini = train(model_mini, lr=0.001, epochs=300, use_full_batch=False)
print(f"Mini-batch: Final loss = {losses_mini[-1]:.6f}")


# ============================================
# Step 7: Plot all results
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Learning rate comparison
for lr, losses in results_lr.items():
    axes[0, 0].plot(losses, label=f'lr={lr}', alpha=0.7)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Effect of Learning Rate')
axes[0, 0].legend()
axes[0, 0].set_yscale('log')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Epochs comparison
colors = ['red', 'orange', 'green', 'blue']
for (epochs, losses), color in zip(results_epochs.items(), colors):
    axes[0, 1].plot(losses, label=f'{epochs} epochs', color=color, alpha=0.7)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Effect of Number of Epochs')
axes[0, 1].legend()
axes[0, 1].set_yscale('log')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Full batch vs mini-batch
axes[1, 0].plot(losses_full, 'b-', label='Full Batch', alpha=0.7)
axes[1, 0].plot(losses_mini, 'r-', label='Mini-batch', alpha=0.5)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('Full Batch vs Mini-batch')
axes[1, 0].legend()
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Final prediction with best settings
torch.manual_seed(42)
model_best = SimpleNetwork()
train(model_best, lr=0.001, epochs=1000)

with torch.no_grad():
    pred_best = model_best(x_data).numpy()

axes[1, 1].plot(x_data.numpy(), y_data.numpy(), 'b-', linewidth=2, label='Exact')
axes[1, 1].plot(x_data.numpy(), pred_best, 'r--', linewidth=2, label='Predicted')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('y')
axes[1, 1].set_title('Best Result (lr=0.001, epochs=1000)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exercise3_results.png', dpi=150)
plt.show()

print("\n" + "="*50)
print("KEY OBSERVATIONS")
print("="*50)
print("1. Learning rate too high (0.1) -> training may diverge")
print("2. Learning rate too low (0.0001) -> very slow convergence")
print("3. Good learning rate (0.001) -> smooth, fast convergence")
print("4. More epochs -> lower loss (but diminishing returns)")
print("5. Full batch -> smoother loss curve")
print("6. Mini-batch -> noisier but sometimes faster")
print("\nPlot saved as 'exercise3_results.png'")
