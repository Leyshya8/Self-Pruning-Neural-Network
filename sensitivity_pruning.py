import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
import os

# CORE: SensitivityLinear Layer
class SensitivityLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, ema_decay: float = 0.95):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.ema_decay    = ema_decay

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        self.register_buffer("mask",     torch.ones(out_features, in_features))
        self.register_buffer("grad_ema", torch.zeros(out_features, in_features))

        self._register_grad_hook()

    def _register_grad_hook(self):
        def hook(grad):
            self.grad_ema.mul_(self.ema_decay).add_(
                grad.detach().abs() * (1.0 - self.ema_decay)
            )
            return grad  
        self.weight.register_hook(hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        effective_weight = self.weight * self.mask
        return F.linear(x, effective_weight, self.bias)
 

    def sensitivity_scores(self) -> torch.Tensor:
        scores = self.weight.detach().abs() * self.grad_ema
        return scores * self.mask  

    def prune_to_sparsity(self, target_sparsity: float):
        scores = self.sensitivity_scores()
        total  = scores.numel()
        n_prune = int(total * target_sparsity)
        if n_prune == 0:
            return

        # Flatten, find the threshold value at the n_prune-th smallest score
        flat_scores = scores.view(-1)
        threshold   = flat_scores.kthvalue(n_prune).values.item()

        # New mask: keep weights whose score is above threshold
        # (ties broken conservatively: keep if >= threshold)
        new_mask = (scores > threshold).float()
        self.mask.copy_(new_mask)

        # Zero out the weight values that are pruned so they don't accumulate
        # momentum in the optimizer for nothing
        with torch.no_grad():
            self.weight.mul_(self.mask)

    @property
    def sparsity(self) -> float:
        """Fraction of weights currently masked out."""
        return 1.0 - self.mask.mean().item()

# Network

class SensitivityPruningNet(nn.Module):
    def __init__(self, hidden_dims=(512, 256, 128), ema_decay: float = 0.95):
        super().__init__()
        dims = [3072] + list(hidden_dims) + [10]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(SensitivityLinear(dims[i], dims[i + 1], ema_decay=ema_decay))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(0.3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))

    def sensitivity_layers(self):
        return [m for m in self.modules() if isinstance(m, SensitivityLinear)]

    def prune_all(self, target_sparsity: float):
        for layer in self.sensitivity_layers():
            layer.prune_to_sparsity(target_sparsity)

    def overall_sparsity(self) -> float:
        pruned = total = 0
        for m in self.sensitivity_layers():
            pruned += (m.mask == 0).sum().item()
            total  += m.mask.numel()
        return pruned / total if total > 0 else 0.0

    def all_scores(self) -> np.ndarray:
        parts = [m.sensitivity_scores().cpu().numpy().ravel()
                 for m in self.sensitivity_layers()]
        return np.concatenate(parts)

    def all_mask_values(self) -> np.ndarray:
        parts = [m.mask.cpu().numpy().ravel() for m in self.sensitivity_layers()]
        return np.concatenate(parts)



class ProgressivePruningSchedule:
    def __init__(
        self,
        final_sparsity: float = 0.80,
        start_sparsity: float = 0.10,
        prune_every:    int   = 3,
        n_steps:        int   = 8,
        recovery_epochs:int   = 1,
    ):
        self.final_sparsity   = final_sparsity
        self.start_sparsity   = start_sparsity
        self.prune_every      = prune_every
        self.n_steps          = n_steps
        self.recovery_epochs  = recovery_epochs
        self.step_count       = 0
        self.last_prune_epoch = -999

    def should_prune(self, epoch: int) -> bool:
        in_recovery = (epoch - self.last_prune_epoch) <= self.recovery_epochs
        due_for_prune = (epoch % self.prune_every == 0)
        return due_for_prune and not in_recovery and self.step_count < self.n_steps

    def current_target_sparsity(self) -> float:
        t = min(self.step_count / max(self.n_steps - 1, 1), 1.0)
        # Cubic ease-in: slow start, faster middle
        sparsity = self.start_sparsity + (self.final_sparsity - self.start_sparsity) * (t ** 3)
        return sparsity

    def record_prune(self, epoch: int):
        self.last_prune_epoch = epoch
        self.step_count      += 1

    def __repr__(self):
        return (f"ProgressivePruningSchedule("
                f"step={self.step_count}/{self.n_steps}, "
                f"target={self.current_target_sparsity():.1%})")


# Data

def get_loaders(batch_size=128, num_workers=2):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
    test_set  = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
    return (DataLoader(train_set, batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
            DataLoader(test_set,  batch_size, shuffle=False, num_workers=num_workers, pin_memory=True))


# Training

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = correct = seen = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = F.cross_entropy(logits, labels)
        loss.backward()           # <-- grad hook fires here, updating grad_ema
        optimizer.step()

        # Re-zero pruned weights (optimizer may have nudged them via momentum)
        for m in model.sensitivity_layers():
            with torch.no_grad():
                m.weight.mul_(m.mask)

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        seen       += imgs.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = seen = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        seen    += imgs.size(0)
    return correct / seen



# Full Experiment

def run_experiment(
    final_sparsity: float,
    train_loader,
    test_loader,
    epochs:         int   = 40,
    device:         str   = "cpu",
    label:          str   = "",
):
    print(f"\n{'='*60}")
    print(f"  Target sparsity={final_sparsity:.0%}  |  {label}")
    print(f"{'='*60}")

    model    = SensitivityPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    pruner = ProgressivePruningSchedule(
        final_sparsity  = final_sparsity,
        start_sparsity  = 0.10,
        prune_every     = 4,
        n_steps         = 6,
        recovery_epochs = 1,
    )

    history = defaultdict(list)

    for epoch in range(1, epochs + 1):

        if pruner.should_prune(epoch):
            target = pruner.current_target_sparsity()
            model.prune_all(target)
            pruner.record_prune(epoch)
            print(f"  [Prune @ epoch {epoch:3d}]  target={target:.1%}  "
                  f"actual={model.overall_sparsity():.1%}")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        sparsity = model.overall_sparsity()
        history["sparsity"].append(sparsity)
        history["tr_acc"].append(tr_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={tr_loss:.4f} | "
                  f"tr_acc={tr_acc:.3f} | sparsity={sparsity:.1%}")

    test_acc = evaluate(model, test_loader, device)
    sparsity  = model.overall_sparsity()
    scores    = model.all_scores()
    masks     = model.all_mask_values()

    print(f"\n  ➤ Test Accuracy  : {test_acc*100:.2f}%")
    print(f"  ➤ Final Sparsity : {sparsity*100:.2f}%")

    return {
        "label":        label,
        "final_sparsity": final_sparsity,
        "test_acc":     test_acc,
        "sparsity":     sparsity,
        "scores":       scores,
        "masks":        masks,
        "history":      history,
    }


# Plotting

def plot_all(results: list[dict], save_path="sensitivity_results.png"):
    n   = len(results)
    fig = plt.figure(figsize=(7 * n, 11))
    fig.suptitle(
        "Magnitude × Gradient Sensitivity Pruning\n"
        "Top row: sensitivity score distribution  |  Bottom row: training curves",
        fontsize=13, fontweight="bold", y=1.01
    )
    gs = gridspec.GridSpec(2, n, figure=fig, hspace=0.45, wspace=0.35)

    for col, res in enumerate(results):
        ax0 = fig.add_subplot(gs[0, col])
        scores = res["scores"]
        masks  = res["masks"]

        # Separate pruned vs active scores
        active_scores = scores[masks == 1]
        pruned_scores = scores[masks == 0]

        ax0.hist(pruned_scores, bins=60, color="#E57373", alpha=0.7, label="pruned", density=True)
        ax0.hist(active_scores, bins=60, color="#42A5F5", alpha=0.7, label="active", density=True)
        ax0.set_title(
            f"{res['label']}\nAcc={res['test_acc']*100:.1f}%  "
            f"Sparse={res['sparsity']*100:.1f}%",
            fontsize=10
        )
        ax0.set_xlabel("Sensitivity score  (|w| × |grad EMA|)")
        ax0.set_ylabel("Density")
        ax0.legend(fontsize=9)
        ax0.set_yscale("log")
        ax0.grid(axis="y", alpha=0.3)

        ax1 = fig.add_subplot(gs[1, col])
        hist = res["history"]
        epochs = range(1, len(hist["sparsity"]) + 1)

        ax1_r = ax1.twinx()
        ax1.plot(epochs, hist["tr_acc"],   color="#42A5F5", linewidth=1.5, label="Train acc")
        ax1_r.plot(epochs, hist["sparsity"], color="#EF9F27", linewidth=1.5, linestyle="--", label="Sparsity")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train accuracy", color="#42A5F5")
        ax1_r.set_ylabel("Sparsity", color="#EF9F27")
        ax1.set_ylim(0, 1)
        ax1_r.set_ylim(0, 1)
        ax1.grid(alpha=0.2)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_r.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[Plot saved → {save_path}]")
    return fig


def print_table(results):
    print("\n" + "─" * 62)
    print(f"  {'Experiment':<20}  {'Test Acc':>10}  {'Sparsity':>10}  {'Target':>8}")
    print("─" * 62)
    for r in results:
        print(f"  {r['label']:<20}  {r['test_acc']*100:>9.2f}%  "
              f"{r['sparsity']*100:>9.2f}%  {r['final_sparsity']*100:>7.0f}%")
    print("─" * 62)



if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 40
    print(f"Device: {DEVICE}")

    train_loader, test_loader = get_loaders()

    experiments = [
        dict(final_sparsity=0.50, label="50% target  (light)"),
        dict(final_sparsity=0.75, label="75% target  (medium)"),
        dict(final_sparsity=0.90, label="90% target  (aggressive)"),
    ]

    results = []
    for exp in experiments:
        res = run_experiment(
            **exp,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=EPOCHS,
            device=DEVICE,
        )
        results.append(res)

    print_table(results)
    plot_all(results)
    plt.show()
