#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm


CLASS_BLOCKED = 0
CLASS_NONE = 1
CLASS_OPEN = 2
CLASS_NAMES = ["blocked", "none", "open"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0


def build_model(pretrained=True):
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=pretrained)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
    return torch.device(name)


def make_dataset(dataset_dir, image_size):
    counts = raw_class_counts(Path(dataset_dir))
    missing = [name for name, count in counts.items() if count <= 0]
    if missing:
        raise RuntimeError(
            f"Need at least one image for every shortcut class before training. "
            f"Missing={missing}, counts={counts}"
        )

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return datasets.ImageFolder(dataset_dir, transform=transform)


def split_dataset(dataset, test_percent):
    num_test = int(test_percent * len(dataset))
    if len(dataset) > 1:
        num_test = max(1, min(num_test, len(dataset) - 1))
    else:
        num_test = 0
    return torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])


def raw_class_counts(dataset_dir):
    counts = {}
    for class_name in CLASS_NAMES:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            counts[class_name] = 0
            continue
        counts[class_name] = sum(
            1
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return counts


def class_counts(dataset):
    counts = {name: 0 for name in dataset.classes}
    for target in dataset.targets:
        counts[dataset.classes[int(target)]] += 1
    return counts


def evaluate(model, loader, device, loss_fn):
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    confusion = torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            batch_size = labels.size(0)
            loss_sum += float(loss) * batch_size
            correct += int((preds == labels).sum())
            count += batch_size

            for target, pred in zip(labels.cpu(), preds.cpu()):
                confusion[int(target), int(pred)] += 1

    avg_loss = loss_sum / max(count, 1)
    acc = 100.0 * correct / max(count, 1)
    return avg_loss, acc, confusion


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet18 shortcut none/blocked/open classifier")
    parser.add_argument("--dataset-dir", default="Shortcut/shortcut_detect/dataset_bev")
    parser.add_argument("--best-model-path", default="Shortcut/shortcut_detect/best_shortcut_resnet18.pth")
    parser.add_argument("--last-model-path", default="Shortcut/shortcut_detect/last_shortcut_resnet18.pth")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-percent", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = make_dataset(args.dataset_dir, args.image_size)
    if len(dataset) == 0:
        raise RuntimeError(f"No images found in dataset directory: {args.dataset_dir}")

    expected = {
        "blocked": CLASS_BLOCKED,
        "none": CLASS_NONE,
        "open": CLASS_OPEN,
    }
    if dataset.class_to_idx != expected:
        raise RuntimeError(
            f"Expected class mapping {expected}, got {dataset.class_to_idx}. "
            "Use dataset_bev/blocked, dataset_bev/none and dataset_bev/open directories."
        )
    counts = class_counts(dataset)
    missing = [name for name, count in counts.items() if count <= 0]
    if missing:
        raise RuntimeError(
            f"Need at least one image for every shortcut class. "
            f"Missing={missing}, counts={counts}"
        )

    train_dataset, test_dataset = split_dataset(dataset, args.test_percent)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = resolve_device(args.device)
    model = build_model(pretrained=not args.no_pretrained).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    Path(args.best_model_path).parent.mkdir(parents=True, exist_ok=True)
    if args.last_model_path:
        Path(args.last_model_path).parent.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    best_loss = 1e9

    print(f"dataset: {len(dataset)} images classes={dataset.class_to_idx} counts={counts}")
    print(f"train/test: {len(train_dataset)}/{len(test_dataset)}")
    print(f"device: {device}")
    print(f"best model path: {args.best_model_path}")
    print(f"last model path: {args.last_model_path}")
    print("label: blocked=0, none=1, open=2")

    for epoch in tqdm(range(args.epochs)):
        model.train()
        train_loss = 0.0
        train_count = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            train_loss += float(loss) * batch_size
            train_count += batch_size

        train_loss /= max(train_count, 1)
        test_loss, test_acc, confusion = evaluate(model, test_loader, device, loss_fn)

        if args.last_model_path:
            torch.save(model.state_dict(), args.last_model_path)

        improved = test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss)
        if improved:
            best_acc = test_acc
            best_loss = test_loss
            torch.save(model.state_dict(), args.best_model_path)
            saved = " saved_best"
        else:
            saved = ""

        print(
            f"epoch={epoch + 1:03d} train_loss={train_loss:.6f} "
            f"test_loss={test_loss:.6f} test_acc={test_acc:.2f}% "
            f"lr={get_lr(optimizer):.6f}{saved}"
        )
        print(f"confusion rows=true cols=pred classes={CLASS_NAMES}: {confusion.tolist()}")


if __name__ == "__main__":
    main()
