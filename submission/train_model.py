"""
Step 1: Train a small CNN on MNIST and export to ONNX format.

Architecture (deliberately small for feasible ZK proving):
- Conv2d(1, 4, 3, stride=2, padding=1) + ReLU    -> 28x28 becomes 14x14
- Conv2d(4, 8, 3, stride=2, padding=1) + ReLU     -> 14x14 becomes 7x7
- Flatten(8 * 7 * 7 = 392) -> Linear(392, 10)

Total parameters: ~4,300
Why so small? Because EZKL must convert every single operation into
arithmetic circuit gates. Fewer parameters = smaller circuit = faster proving.
A MobileNet (~3M params) needs 139 GB RAM to prove. This model needs under 1 GB.

Training: ~2 minutes on GPU, ~5 minutes on CPU. Accuracy: 97-98%.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os
import json
import numpy as np


# ===================== MODEL DEFINITION =====================

class SmallCNN(nn.Module):
    """
    A minimal CNN designed for two goals:
    1. Reasonable accuracy on MNIST (97%+)
    2. Small enough for EZKL to compile into a ZK circuit

    Design choices:
    - Strided convolutions instead of MaxPool (MaxPool works in EZKL,
      but strided conv produces a simpler, more predictable circuit)
    - ReLU activation (well-supported in EZKL via lookup tables)
    - No batch normalization (it complicates quantization because it
      involves running statistics that change between training and inference)
    - No dropout (not needed at inference time, and we export inference-mode)
    """

    def __init__(self):
        super(SmallCNN, self).__init__()
        # Layer 1: 1 input channel, 4 filters, 3x3 kernel, stride 2
        # Input: (1, 28, 28) -> Output: (4, 14, 14)
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()

        # Layer 2: 4 input channels, 8 filters, 3x3 kernel, stride 2
        # Input: (4, 14, 14) -> Output: (8, 7, 7)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()

        # Layer 3: Fully connected, 8*7*7=392 inputs, 10 outputs (digits 0-9)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 7 * 7, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x


# ===================== TRAINING =====================

def train_model():
    # Force CPU: the RTX 5070 (sm_120 / Blackwell) is not yet supported
    # by stable PyTorch. Our model is tiny (4K params), so CPU is fine.
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data loading with minimal transforms (just normalize)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # MNIST pixel values are 0-1 after ToTensor.
        # We normalize to mean=0.5, std=0.5 so values are in [-1, 1].
        # This bounded range helps quantization later.
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Downloading MNIST dataset (first time only)...")
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Larger batch size = fewer steps per epoch = faster training
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)

    # Initialize model
    model = SmallCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Optimizer: Adam with a relatively high learning rate since the model is small
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    # Train for only 5 epochs (enough for 97%+ on MNIST with this setup)
    num_epochs = 5
    print(f"\nTraining for {num_epochs} epochs...")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.1f} seconds")

    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100.0 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")

    # ===================== SAVE MODEL =====================
    model_path = os.path.join("models", "mnist_cnn.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nPyTorch model saved to: {model_path}")

    # ===================== EXPORT TO ONNX =====================
    # EZKL needs the model in ONNX format.
    # We create a dummy input matching MNIST dimensions: batch=1, channels=1, height=28, width=28
    model.cpu()
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28)

    onnx_path = os.path.join("models", "mnist_cnn.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,           # store trained weights inside the ONNX file
        opset_version=11,             # ONNX opset version (11 is widely supported by EZKL)
        do_constant_folding=True,     # optimize by folding constant operations
        input_names=["input"],        # name for the input tensor
        output_names=["output"],      # name for the output tensor
        dynamic_axes=None             # fixed input size (EZKL needs static shapes)
    )
    print(f"ONNX model exported to: {onnx_path}")

    # ===================== SAVE A TEST SAMPLE =====================
    # Pick one test image and save it as JSON for the proving step.
    # EZKL expects: {"input_data": [[flat_val1, flat_val2, ..., flat_val784]]}
    # Each input tensor must be flattened into a 1D list, wrapped in a list.
    test_image, test_label = test_dataset[0]  # first test image

    # Flatten the [1, 1, 28, 28] tensor into [784] values
    flat_input = test_image.unsqueeze(0).reshape(-1).numpy().tolist()

    # Run inference to get the expected output (for verification later)
    with torch.no_grad():
        expected_output = model(test_image.unsqueeze(0)).numpy().tolist()

    # EZKL input format: {"input_data": [array_per_input_tensor]}
    input_json = {"input_data": [flat_input]}
    input_json_path = os.path.join("models", "test_input.json")
    with open(input_json_path, "w") as f:
        json.dump(input_json, f)
    print(f"Test input saved to: {input_json_path}")
    print(f"Test image label: {test_label}")
    print(f"Model prediction: {np.argmax(expected_output)}")

    # Save metadata for later scripts
    metadata = {
        "total_params": total_params,
        "test_accuracy": test_acc,
        "training_time_seconds": round(train_time, 1),
        "test_label": int(test_label),
        "model_prediction": int(np.argmax(expected_output)),
        "onnx_path": onnx_path,
        "input_json_path": input_json_path,
    }
    with open(os.path.join("models", "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved. Ready for EZKL pipeline.")


if __name__ == "__main__":
    train_model()