"""
Exercise 2: SGD vs Adam Optimizer Comparison
=============================================
Compare how SGD and Adam optimizers train on the same problem

Run: python exercise2_optimizers.py
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

        # Use Xavier initialization for fair comparison
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
def train(model, optimizer, epochs=1000):
    """Train and return loss history"""
    loss_fn = nn.MSELoss()
    losses = []

    for epoch in range(epochs):
        prediction = model(x_data)
        loss = loss_fn(prediction, y_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.6f}")

    return losses


# ============================================
# Step 4: Train with SGD
# ============================================
print("="*50)
print("Training with SGD (lr=0.01)")
print("="*50)

# Set random seed for reproducibility
torch.manual_seed(42)
model_sgd = SimpleNetwork()
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
losses_sgd = train(model_sgd, optimizer_sgd, epochs=1000)


# ============================================
# Step 5: Train with Adam
# ============================================
print("\n" + "="*50)
print("Training with Adam (lr=0.001)")
print("="*50)

# Reset seed for fair comparison
torch.manual_seed(42)
model_adam = SimpleNetwork()
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
losses_adam = train(model_adam, optimizer_adam, epochs=1000)


# ============================================
# Step 6: Plot results
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Plot 1: Loss comparison
axes[0].plot(losses_sgd, 'r-', label='SGD (lr=0.01)', alpha=0.7)
axes[0].plot(losses_adam, 'b-', label='Adam (lr=0.001)', alpha=0.7)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('Training Loss: SGD vs Adam')
axes[0].legend()
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

# Plot 2: SGD predictions
with torch.no_grad():
    pred_sgd = model_sgd(x_data).numpy()
axes[1].plot(x_data.numpy(), y_data.numpy(), 'b-', linewidth=2, label='Exact')
axes[1].plot(x_data.numpy(), pred_sgd, 'r--', linewidth=2, label='SGD')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('SGD Result')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Adam predictions
with torch.no_grad():
    pred_adam = model_adam(x_data).numpy()
axes[2].plot(x_data.numpy(), y_data.numpy(), 'b-', linewidth=2, label='Exact')
axes[2].plot(x_data.numpy(), pred_adam, 'g--', linewidth=2, label='Adam')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
axes[2].set_title('Adam Result')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exercise2_results.png', dpi=150)
plt.show()

print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)
print(f"SGD  - Final Loss: {losses_sgd[-1]:.6f}")
print(f"Adam - Final Loss: {losses_adam[-1]:.6f}")
print(f"\nAdam typically converges faster and to a better solution!")
print("Plot saved as 'exercise2_results.png'")
