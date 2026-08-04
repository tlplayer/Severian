#!/usr/bin/env python3
import torch


weights = torch.tensor([[1.0, 0.5], [-1.0, 2.0]], dtype=torch.float64)
hidden = torch.tensor([[2.0], [-1.0]], dtype=torch.float64)
bias = torch.tensor([[0.25], [0.5]], dtype=torch.float64)
result = torch.relu(weights @ hidden + bias).flatten().tolist()
print(f"[{result[0]:g}, {result[1]:g}]")
