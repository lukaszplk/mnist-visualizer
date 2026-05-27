# mnist-visualizer

[![CI](https://github.com/lukaszplk/mnist-visualizer/actions/workflows/ci.yml/badge.svg)](https://github.com/lukaszplk/mnist-visualizer/actions/workflows/ci.yml)

A live neural network training visualizer for MNIST digit recognition (0–9).

Watch a fully-connected MLP learn in real time — node colours show activations, edge colours show weights, and the stats dashboard tracks loss, accuracy, and per-layer health every batch.

![screenshot placeholder](assets/screenshot.png)

## Features

- **Live network graph** — nodes coloured by activation value (blue→white→red), edges coloured by weight magnitude and sign (green/red)
- **Real-time stats** — loss curve, batch/epoch/validation accuracy curves updated every batch
- **Per-layer table** — activation mean/std, dead neuron %, weight mean, gradient mean for each layer
- **Pause / Resume / Stop** controls
- Runs on CPU or GPU automatically

## Architecture

```
Input (784) → Hidden 1 (128, ReLU) → Hidden 2 (64, ReLU) → Output (10)
```

Trained with Adam on MNIST, batch size 64, 20 epochs.

## Install

```bash
git clone https://github.com/lukaszplk/mnist-visualizer
cd mnist-visualizer
pip install -e .
```

> MNIST data (~11 MB) is downloaded automatically on first run to `~/.cache/mnist_visualizer/`.

## Run

```bash
mnist-visualizer
```

Or directly:
```bash
python -m mnist_visualizer.app
```

## Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  MNIST Neural Network Visualizer                        [Pause]  │
├────────────────────────┬─────────────────────────────────────────┤
│  Network Graph         │  Training Statistics                    │
│                        │  Epoch: 3/20  Batch: 45/937            │
│  ●━━━━●━━━━●━━━━●      │  Loss: 0.1243   Batch acc: 96.9%       │
│  ●    ●    ●    ●      │  Val accuracy: 97.84%                   │
│  ●    ●    ●           │  ┌─────────────────────────────────┐   │
│  …    …    …           │  │ Loss curve                      │   │
│                        │  ├─────────────────────────────────┤   │
│  node  = activation    │  │ Accuracy (batch / epoch / val)  │   │
│  edge  = weight        │  ├─────────────────────────────────┤   │
│                        │  │ Per-layer: Act μ σ  Dead%  ∇W   │   │
│                        │  └─────────────────────────────────┘   │
└────────────────────────┴─────────────────────────────────────────┘
```

## Development

```bash
pip install -e .
python -m mnist_visualizer.app
```
