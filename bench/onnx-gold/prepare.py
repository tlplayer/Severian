#!/usr/bin/env python3
"""Train an Iris MLP, export ONNX, and generate equivalent Severian source."""

from __future__ import annotations

import json
from pathlib import Path
import urllib.request

import numpy as np
import onnx
from onnx import numpy_helper
import torch


HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
LABELS = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}


class IrisMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(4, 12)
        self.output = torch.nn.Linear(12, 3)

    def forward(self, values):
        return self.output(torch.relu(self.hidden(values)))


def load_iris():
    dataset = GENERATED / "iris.data"
    if not dataset.exists():
        dataset.write_bytes(urllib.request.urlopen(DATA_URL, timeout=30).read())
    rows = [line.split(",") for line in dataset.read_text().splitlines() if line]
    features = np.asarray(
        [[float(value) for value in row[:4]] for row in rows], dtype=np.float32
    )
    targets = np.asarray([LABELS[row[4]] for row in rows], dtype=np.int64)
    mean = features.mean(axis=0)
    deviation = features.std(axis=0)
    return (features - mean) / deviation, targets


def float_literal(value) -> str:
    literal = format(float(np.float32(value)), ".9g")
    if "." not in literal and "e" not in literal:
        literal += ".0"
    return literal


def list_literal(values) -> str:
    flattened = np.asarray(values).reshape(-1)
    return "[" + ", ".join(float_literal(value) for value in flattened) + "]"


def generate_severian(model_path: Path, features: np.ndarray):
    model = onnx.load(model_path)
    initializers = {
        item.name: numpy_helper.to_array(item) for item in model.graph.initializer
    }
    hidden_weights = initializers["hidden.weight"]
    hidden_bias = initializers["hidden.bias"]
    output_weights = initializers["output.weight"]
    output_bias = initializers["output.bias"]

    source = f'''import distributed
import tensor

def appendValues(target: list[float], source: list[float]):
    for value in source:
        target.append(value)

def repeatInputs(base: list[float], repeats: int) -> list[float]:
    inputs := []
    for _ in range(0, repeats):
        appendValues(inputs, base)
    return inputs

def inferChunk(
    inputs: list[float],
    start: int,
    end: int,
    hiddenWeights: list[float],
    hiddenBias: list[float],
    outputWeights: list[float],
    outputBias: list[float],
) -> list[float]:
    logits := []
    for sample in range(start, end):
        offset = sample * 4
        features = [inputs[offset], inputs[offset + 1], inputs[offset + 2], inputs[offset + 3]]
        hidden = relu(add(matVec(hiddenWeights, 12, 4, features), hiddenBias))
        appendValues(logits, add(matVec(outputWeights, 3, 12, hidden), outputBias))
    return logits

def inferBatch(
    inputs: list[float],
    samples: int,
    workers: int,
    hiddenWeights: list[float],
    hiddenBias: list[float],
    outputWeights: list[float],
    outputBias: list[float],
) -> list[float]:
    with self and local:
        first = async inferChunk(inputs, shardStart(samples, workers, 0), shardEnd(samples, workers, 0), hiddenWeights, hiddenBias, outputWeights, outputBias)
        second = async inferChunk(inputs, shardStart(samples, workers, 1), shardEnd(samples, workers, 1), hiddenWeights, hiddenBias, outputWeights, outputBias)
        third = async inferChunk(inputs, shardStart(samples, workers, 2), shardEnd(samples, workers, 2), hiddenWeights, hiddenBias, outputWeights, outputBias)
        fourth = async inferChunk(inputs, shardStart(samples, workers, 3), shardEnd(samples, workers, 3), hiddenWeights, hiddenBias, outputWeights, outputBias)
        firstValues = await first
        secondValues = await second
        thirdValues = await third
        fourthValues = await fourth
        logits := []
        appendValues(logits, firstValues)
        appendValues(logits, secondValues)
        appendValues(logits, thirdValues)
        appendValues(logits, fourthValues)
        return logits

def classSum(logits: list[float], target: int) -> float:
    total := 0.0
    samples = size(logits) / 3
    for sample in range(0, samples):
        total += logits[sample * 3 + target]
    return total

def classCount(logits: list[float], target: int) -> int:
    count := 0
    samples = size(logits) / 3
    for sample in range(0, samples):
        offset = sample * 3
        best := 0
        bestValue := logits[offset]
        if logits[offset + 1] > bestValue:
            best = 1
            bestValue = logits[offset + 1]
        if logits[offset + 2] > bestValue:
            best = 2
        if best == target:
            count += 1
    return count

def main():
    baseInputs = {list_literal(features)}
    hiddenWeights = {list_literal(hidden_weights)}
    hiddenBias = {list_literal(hidden_bias)}
    outputWeights = {list_literal(output_weights)}
    outputBias = {list_literal(output_bias)}
    repeats = 400
    inputs = repeatInputs(baseInputs, repeats)
    samples = 150 * repeats
    logits = inferBatch(inputs, samples, 4, hiddenWeights, hiddenBias, outputWeights, outputBias)
    print(size(logits))
    print(classSum(logits, 0))
    print(classSum(logits, 1))
    print(classSum(logits, 2))
    print(classCount(logits, 0))
    print(classCount(logits, 1))
    print(classCount(logits, 2))
'''
    (GENERATED / "model.sev").write_text(source)
    sequential = source.replace(
        "logits = inferBatch(inputs, samples, 4, hiddenWeights, hiddenBias, outputWeights, outputBias)",
        "logits = inferChunk(inputs, 0, samples, hiddenWeights, hiddenBias, outputWeights, outputBias)",
    )
    (GENERATED / "model-sequential.sev").write_text(sequential)


def main():
    GENERATED.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    features, targets = load_iris()
    values = torch.from_numpy(features)
    labels = torch.from_numpy(targets)
    model = IrisMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(600):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(values), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        accuracy = (model(values).argmax(dim=1) == labels).float().mean().item()

    model_path = GENERATED / "iris-mlp.onnx"
    torch.onnx.export(
        model,
        (values,),
        model_path,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    checked = onnx.load(model_path)
    onnx.checker.check_model(checked)
    operators = [node.op_type for node in checked.graph.node]
    if operators != ["Gemm", "Relu", "Gemm"]:
        raise RuntimeError(f"unexpected ONNX graph: {operators}")
    np.save(GENERATED / "features.npy", features)
    generate_severian(model_path, features)
    (GENERATED / "metadata.json").write_text(
        json.dumps({"accuracy": accuracy, "operators": operators}, indent=2) + "\n"
    )
    print(f"trained Iris MLP accuracy: {accuracy:.3%}")
    print(f"ONNX operators: {', '.join(operators)}")
    print(f"generated: {GENERATED / 'model.sev'}")


if __name__ == "__main__":
    main()
