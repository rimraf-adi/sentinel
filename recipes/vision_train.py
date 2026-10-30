"""
Synthetic vision classification smoke test runner for Sentinel.
Runs standalone on CPU or GPU without external network calls.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def run_smoke_test(epochs=2, batch_size=8, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Sentinel Smoke Test] Device: {device}, Epochs: {epochs}, Batch size: {batch_size}")

    model = SimpleConvNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Synthetic data: 32 samples of 3x32x32 images
    torch.manual_seed(42)
    inputs = torch.randn(32, 3, 32, 32).to(device)
    targets = torch.randint(0, 10, (32,)).to(device)

    initial_loss = None
    final_loss = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = len(inputs) // batch_size

        for i in range(num_batches):
            batch_x = inputs[i * batch_size : (i + 1) * batch_size]
            batch_y = targets[i * batch_size : (i + 1) * batch_size]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"[Epoch {epoch + 1}/{epochs}] Loss: {avg_loss:.4f}")

        if initial_loss is None:
            initial_loss = avg_loss
        final_loss = avg_loss

    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
    print(f"[Sentinel Smoke Test] SUCCESS: Loss decreased from {initial_loss:.4f} to {final_loss:.4f}")

if __name__ == "__main__":
    run_smoke_test()
