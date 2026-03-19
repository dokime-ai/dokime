"""Toy world model: predict next state from current state + action.

This is a learning exercise for PyTorch fundamentals.
Trains on DINO-WM's point_maze data (2000 trajectories, 100 steps each).

Usage:
    python train.py

No GPU needed — trains in seconds on CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path


# ── 1. Dataset ───────────────────────────────────────────────────────────────

class TransitionDataset(Dataset):
    """Dataset of (state, action) -> next_state transitions.

    Each sample is one step from a trajectory:
        input:  state_t (4D) concatenated with action_t (2D) = 6D
        target: state_{t+1} (4D)
    """

    def __init__(self, data_dir: str):
        data_dir = Path(data_dir)

        # Load raw data
        states = torch.load(data_dir / "states.pth", weights_only=False)   # (2000, 100, 4)
        actions = torch.load(data_dir / "actions.pth", weights_only=False)  # (2000, 100, 2)

        n_traj, seq_len, state_dim = states.shape
        _, _, action_dim = actions.shape

        # Create transition pairs: (s_t, a_t) -> s_{t+1}
        # For each trajectory, we get seq_len-1 transitions
        inputs = []
        targets = []

        for i in range(n_traj):
            for t in range(seq_len - 1):
                s_t = states[i, t]         # (4,)
                a_t = actions[i, t]        # (2,)
                s_next = states[i, t + 1]  # (4,)

                inputs.append(torch.cat([s_t, a_t]))  # (6,)
                targets.append(s_next)                  # (4,)

        self.inputs = torch.stack(inputs).float()    # (N, 6) — cast to float32
        self.targets = torch.stack(targets).float()  # (N, 4)

        print(f"Loaded {len(self)} transitions from {n_traj} trajectories")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ── 2. Model ─────────────────────────────────────────────────────────────────

class TransitionModel(nn.Module):
    """Simple 2-layer MLP that predicts next state from (state, action).

    Architecture:
        6D input → 128 hidden (ReLU) → 128 hidden (ReLU) → 4D output
    """

    def __init__(self, input_dim=6, hidden_dim=128, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ── 3. Training loop ─────────────────────────────────────────────────────────

def train(
    data_dir: str = "C:/Users/drewm/dino_wm/data/point_maze/point_maze",
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.1,
    save_path: str = "checkpoint.pt",
):
    # Load data
    dataset = TransitionDataset(data_dir)

    # Train/val split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size],
                                       generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    print(f"Train: {train_size}, Val: {val_size}")
    print(f"Batches per epoch: {len(train_loader)}")

    # Model + optimizer
    model = TransitionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print()

    # Training
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            predictions = model(inputs)
            loss = F.mse_loss(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(inputs)

        train_loss /= train_size

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                predictions = model(inputs)
                loss = F.mse_loss(predictions, targets)
                val_loss += loss.item() * len(inputs)

        val_loss /= val_size

        # Report
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, save_path)

        marker = " *" if improved else ""
        print(f"Epoch {epoch+1:3d}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}{marker}")

    print(f"\nBest val_loss: {best_val_loss:.6f}")
    print(f"Checkpoint saved to: {save_path}")

    return model


# ── 4. Evaluation ────────────────────────────────────────────────────────────

def evaluate(model, data_dir: str = "C:/Users/drewm/dino_wm/data/point_maze/point_maze"):
    """Evaluate model: compute per-step prediction error across trajectories."""
    import numpy as np

    states = torch.load(Path(data_dir) / "states.pth", weights_only=False)
    actions = torch.load(Path(data_dir) / "actions.pth", weights_only=False)

    model.eval()
    errors = []

    with torch.no_grad():
        for i in range(min(100, len(states))):  # first 100 trajectories
            traj_error = []
            for t in range(states.shape[1] - 1):
                inp = torch.cat([states[i, t], actions[i, t]]).float().unsqueeze(0)
                pred = model(inp).squeeze(0)
                actual = states[i, t + 1]
                err = (pred - actual).pow(2).sum().sqrt().item()
                traj_error.append(err)
            errors.append(np.mean(traj_error))

    errors = np.array(errors)
    print(f"\nEvaluation (100 trajectories):")
    print(f"  Mean per-step error: {errors.mean():.4f}")
    print(f"  Std:                 {errors.std():.4f}")
    print(f"  Min:                 {errors.min():.4f}")
    print(f"  Max:                 {errors.max():.4f}")


if __name__ == "__main__":
    model = train()
    evaluate(model)
