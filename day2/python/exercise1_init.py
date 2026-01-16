"""
Exercise 1: Effect of Weight Initialization
============================================
Compare random initialization vs Xavier initialization

Run: python exercise1_init.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ============================================
# Step 1: Create the training data
# ============================================
# We want to learn: y = 2x - x^3
x_data = torch.linspace(-2, 2, 101).reshape(-1, 1)  # 101 points
y_data = 2 * x_data - x_data ** 3

print("Data created: 101 points from x=-2 to x=2")
print(f"Target function: y = 2x - x^3\n")


# ============================================
# Step 2: Define the neural network
# ============================================
class SimpleNetwork(nn.Module):
    """A simple 4-layer network: 1 -> 10 -> 20 -> 10 -> 1"""

    def __init__(self, use_xavier=False):
        super().__init__()

        # Define layers
        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 20)
        self.layer3 = nn.Linear(20, 10)
        self.layer4 = nn.Linear(10, 1)

        # Initialize weights
        if use_xavier:
            # Xavier initialization - better for training
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            print("Using Xavier initialization")
        else:
            # Random initialization with large weights - often problematic
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                nn.init.normal_(layer.weight, mean=0, std=1.0)
                nn.init.zeros_(layer.bias)
            print("Using random initialization (std=1.0)")

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        x = self.layer4(x)  # No activation on output
        return x


# ============================================
# Step 3: Training function
# ============================================
def train(model, epochs=500, lr=0.01):
    """Train the model and return loss history"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []

    for epoch in range(epochs):
        # Forward pass
        prediction = model(x_data)
        loss = loss_fn(prediction, y_data)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record loss
        losses.append(loss.item())

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.6f}")

    return losses


# ============================================
# Step 4: Train both models
# ============================================
print("\n" + "="*50)
print("Training Model 1: Random Initialization")
print("="*50)
model_random = SimpleNetwork(use_xavier=False)
losses_random = train(model_random, epochs=500)

print("\n" + "="*50)
print("Training Model 2: Xavier Initialization")
print("="*50)
model_xavier = SimpleNetwork(use_xavier=True)
losses_xavier = train(model_xavier, epochs=500)


# ============================================
# Step 5: Plot results
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Plot 1: Training loss comparison
axes[0].plot(losses_random, 'r-', label='Random Init', alpha=0.7)
axes[0].plot(losses_xavier, 'b-', label='Xavier Init', alpha=0.7)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('Training Loss Comparison')
axes[0].legend()
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

# Plot 2: Random init predictions
with torch.no_grad():
    pred_random = model_random(x_data).numpy()
axes[1].plot(x_data.numpy(), y_data.numpy(), 'b-', linewidth=2, label='Exact')
axes[1].plot(x_data.numpy(), pred_random, 'r--', linewidth=2, label='Predicted')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Random Init Result')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Xavier init predictions
with torch.no_grad():
    pred_xavier = model_xavier(x_data).numpy()
axes[2].plot(x_data.numpy(), y_data.numpy(), 'b-', linewidth=2, label='Exact')
axes[2].plot(x_data.numpy(), pred_xavier, 'g--', linewidth=2, label='Predicted')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
axes[2].set_title('Xavier Init Result')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exercise1_results.png', dpi=150)
plt.show()

print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)
print(f"Random Init - Final Loss: {losses_random[-1]:.6f}")
print(f"Xavier Init - Final Loss: {losses_xavier[-1]:.6f}")
print(f"\nXavier initialization typically trains better!")
print("Plot saved as 'exercise1_results.png'")
