# Toy World Model — PyTorch Learning Project

A simple transition model trained on DINO-WM's point_maze data.
This is what DINO-WM does (without the frozen DINOv2 encoder) — predict
the next state from current state + action.

## Data
- Source: `C:\Users\drewm\dino_wm\data\point_maze\point_maze\`
- States: (2000, 100, 4) — 2000 trajectories, 100 steps, 4D state
- Actions: (2000, 100, 2) — 2D actions
- All trajectories are length 100

## Architecture
- Input: state (4D) + action (2D) = 6D
- Model: 2-layer MLP with ReLU
- Output: predicted next state (4D)
- Loss: MSE between predicted and actual next state

## What You'll Learn
1. torch.utils.data.Dataset (custom dataset class)
2. torch.utils.data.DataLoader (batching, shuffling)
3. nn.Module (model definition)
4. Training loop (forward, loss, backward, step)
5. Train/val split
6. Saving checkpoints
