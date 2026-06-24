#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm


IMAGE_SIZE = 224.0
CAR_X = IMAGE_SIZE / 2.0
CAR_Y = IMAGE_SIZE


def get_label_x(path):
    return float(int(path[3:6]))


def get_label_y(path):
    return float(int(path[7:10]))


def label_to_direction(label_x, label_y):
    dx = float(label_x) - CAR_X
    dy = CAR_Y - float(label_y)
    norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    return dx / norm, dy / norm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0


def normalize_vectors(vectors):
    return vectors / torch.clamp(torch.norm(vectors, dim=1, keepdim=True), min=1e-6)


def weighted_direction_loss(outputs, labels, weights):
    per_sample = F.mse_loss(normalize_vectors(outputs), labels, reduction="none").mean(dim=1)
    return (per_sample * weights).sum() / torch.clamp(weights.sum(), min=1e-6)


def angular_errors_deg(outputs, labels):
    outputs = normalize_vectors(outputs)
    labels = normalize_vectors(labels)
    dots = torch.sum(outputs * labels, dim=1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(dots))


class WeightedDirectionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dir,
        feedback_dir,
        feedback_weight=3.0,
        random_hflips=False,
        rgb_input=False,
    ):
        self.random_hflips = random_hflips
        self.rgb_input = rgb_input
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
        self.samples = []

        self.add_directory(base_dir, weight=1.0, source="base")
        self.add_directory(feedback_dir, weight=float(feedback_weight), source="feedback")

    def add_directory(self, directory, weight, source):
        if not directory:
            return
        directory = str(directory)
        for path in sorted(glob.glob(os.path.join(directory, "*.jpg"))):
            name = os.path.basename(path)
            if not name.startswith("xy_"):
                continue
            self.samples.append((path, float(weight), source))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, weight, source = self.samples[idx]
        image = PIL.Image.open(image_path)
        label_x = get_label_x(os.path.basename(image_path))
        label_y = get_label_y(os.path.basename(image_path))
        vx, vy = label_to_direction(label_x, label_y)

        if float(np.random.rand(1)) > 0.5 and self.random_hflips:
            image = transforms.functional.hflip(image)
            vx = -vx

        image = self.color_jitter(image)
        image = transforms.functional.to_tensor(image)
        if not self.rgb_input:
            image = image.numpy()[::-1].copy()
            image = torch.from_numpy(image)
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image, torch.tensor([vx, vy]).float(), torch.tensor(weight).float(), source


def build_model(initial_model_path=None):
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except AttributeError:
        model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 2)

    if initial_model_path:
        initial_model_path = Path(initial_model_path)
        if initial_model_path.exists():
            state = torch.load(initial_model_path, map_location="cpu")
            model.load_state_dict(state)
            print(f"loaded initial model: {initial_model_path}")
        else:
            print(f"initial model not found, start from ImageNet weights: {initial_model_path}")
    return model


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
    return torch.device(name)


def split_dataset(dataset, test_percent):
    num_test = int(float(test_percent) * len(dataset))
    if len(dataset) > 1:
        num_test = max(1, min(num_test, len(dataset) - 1))
    else:
        num_test = 0
    return torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])


def collate_batch(batch):
    images, labels, weights, sources = zip(*batch)
    return torch.stack(images), torch.stack(labels), torch.stack(weights), sources


def evaluate(model, loader, device, threshold_deg):
    model.eval()
    loss_sum = 0.0
    error_sum = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for images, labels, weights, _sources in loader:
            images = images.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            outputs = model(images)
            loss = weighted_direction_loss(outputs, labels, weights)
            errors = angular_errors_deg(outputs, labels)
            batch_size = labels.size(0)
            loss_sum += float(loss) * batch_size
            error_sum += float(errors.sum())
            correct += int((errors <= threshold_deg).sum())
            count += batch_size
    return (
        loss_sum / max(count, 1),
        100.0 * correct / max(count, 1),
        error_sum / max(count, 1),
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Fine-tune direction model with weighted feedback samples")
    parser.add_argument("--base-dataset-dir", required=True)
    parser.add_argument("--feedback-dataset-dir", required=True)
    parser.add_argument("--feedback-weight", type=float, default=3.0)
    parser.add_argument("--initial-model-path", default="")
    parser.add_argument("--best-model-path", required=True)
    parser.add_argument("--last-model-path", default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-percent", type=float, default=0.1)
    parser.add_argument("--test-acc-threshold-deg", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--random-hflips", action="store_true")
    parser.add_argument("--rgb-input", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    dataset = WeightedDirectionDataset(
        args.base_dataset_dir,
        args.feedback_dataset_dir,
        feedback_weight=args.feedback_weight,
        random_hflips=args.random_hflips,
        rgb_input=args.rgb_input,
    )
    if len(dataset) == 0:
        raise RuntimeError("No xy_*.jpg files found in base/feedback dataset directories")

    base_count = sum(1 for _path, _weight, source in dataset.samples if source == "base")
    feedback_count = sum(1 for _path, _weight, source in dataset.samples if source == "feedback")
    train_dataset, test_dataset = split_dataset(dataset, args.test_percent)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    device = resolve_device(args.device)
    model = build_model(args.initial_model_path).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    Path(args.best_model_path).parent.mkdir(parents=True, exist_ok=True)
    if args.last_model_path:
        Path(args.last_model_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"base/feedback: {base_count}/{feedback_count}")
    print(f"feedback weight: {args.feedback_weight}")
    print(f"train/test: {len(train_dataset)}/{len(test_dataset)}")
    print(f"device: {device}")
    print(f"best model path: {args.best_model_path}")
    print(f"last model path: {args.last_model_path}")

    best_loss = 1e9
    for epoch in tqdm(range(args.epochs)):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for images, labels, weights, _sources in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = weighted_direction_loss(outputs, labels, weights)
            loss.backward()
            optimizer.step()
            batch_size = labels.size(0)
            train_loss_sum += float(loss) * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / max(train_count, 1)
        test_loss, test_acc, test_error_deg = evaluate(
            model,
            test_loader,
            device,
            args.test_acc_threshold_deg,
        )

        if args.last_model_path:
            torch.save(model.state_dict(), args.last_model_path)
        if test_loss < best_loss:
            torch.save(model.state_dict(), args.best_model_path)
            best_loss = test_loss
            saved = " saved_best"
        else:
            saved = ""

        print(
            f"epoch={epoch + 1:03d} train_loss={train_loss:.6f} "
            f"test_loss={test_loss:.6f} test_acc={test_acc:.2f}% "
            f"test_err_deg={test_error_deg:.2f} lr={get_lr(optimizer):.6f}{saved}"
        )


if __name__ == "__main__":
    main()
