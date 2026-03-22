from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.checkpoint import load_checkpoint, save_checkpoint
from src.common.cli import add_common_args
from src.common.device import resolve_device
from src.common.logger import ExperimentLogger
from src.common.paths import make_run_dir, save_args
from src.common.seed import set_seed
from src.dl.data.synthetic import make_classification_dataset
from src.dl.data.vision import make_mnist_datasets
from src.dl.models.mlp import MLPClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny MLP classifier on synthetic or MNIST data.")
    parser = add_common_args(parser, default_exp_name="toy_classifier")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "mnist"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--input-dim", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def prepare_batch(x: torch.Tensor, y: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.to(device)
    y = y.to(device)
    x = x.view(x.size(0), -1)
    return x, y


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = prepare_batch(x, y, device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_count += x.size(0)

    return total_loss / total_count, total_correct / total_count


def build_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset, int, int]:
    if args.dataset == "synthetic":
        dataset = make_classification_dataset(
            num_samples=args.num_samples,
            input_dim=args.input_dim,
            seed=args.seed,
        )
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        return train_set, val_set, args.input_dim, args.num_classes

    train_set, val_set = make_mnist_datasets(data_root=args.data_root)
    return train_set, val_set, 28 * 28, 10


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    run_dir = make_run_dir(task="dl", exp_name=args.exp_name)
    save_args(run_dir, vars(args))
    logger = ExperimentLogger(run_dir)

    train_set, val_set, model_input_dim, num_classes = build_datasets(args)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = MLPClassifier(
        input_dim=model_input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0
    best_val_acc = 0.0

    if args.resume:
        checkpoint = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_count = 0

        for x, y in train_loader:
            x, y = prepare_batch(x, y, device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_correct += (logits.argmax(dim=1) == y).sum().item()
            running_count += x.size(0)

        train_loss = running_loss / running_count
        train_acc = running_correct / running_count
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        logger.log_metrics(
            epoch,
            {
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
            },
        )

        if (epoch + 1) % args.log_every == 0:
            print(
                f"[DL] epoch={epoch + 1}/{args.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        latest_path = run_dir / "checkpoints" / "latest.pt"
        save_checkpoint(
            latest_path,
            {
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
            },
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_path = run_dir / "checkpoints" / "best.pt"
            save_checkpoint(
                best_path,
                {
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "args": vars(args),
                },
            )

    logger.close()
    print(f"[DL] done. output_dir={run_dir}")


if __name__ == "__main__":
    main()
