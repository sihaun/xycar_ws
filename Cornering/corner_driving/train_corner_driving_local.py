#!/usr/bin/env python3

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


def direction_loss(outputs, labels):
    return F.mse_loss(normalize_vectors(outputs), labels)


def angular_errors_deg(outputs, labels):
    outputs = normalize_vectors(outputs)
    labels = normalize_vectors(labels)
    dots = torch.sum(outputs * labels, dim=1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(dots))


class DirectionDataset(torch.utils.data.Dataset):
    def __init__(self, directory, random_hflips=False, rgb_input=False):
        self.directory = directory
        self.random_hflips = random_hflips
        self.rgb_input = rgb_input
        self.image_paths = glob.glob(os.path.join(self.directory, "*.jpg"))
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
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
        return image, torch.tensor([vx, vy]).float()


def build_parser():
    parser = argparse.ArgumentParser(description="Train cornering front-camera direction-vector model")
    parser.add_argument("--dataset-dir", default="Cornering/corner_driving/dataset_xy")
    parser.add_argument("--best-model-path", default="Cornering/corner_driving/best_corner_driving_direction.pth")
    parser.add_argument("--last-model-path", default="Cornering/corner_driving/last_corner_driving_direction.pth")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-percent", type=float, default=0.2)
    parser.add_argument("--test-acc-threshold-deg", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--random-hflips", action="store_true")
    parser.add_argument("--rgb-input", action="store_true")
    return parser


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False")
    return torch.device(name)


def main():
    args = build_parser().parse_args()
    dataset = DirectionDataset(args.dataset_dir, random_hflips=args.random_hflips, rgb_input=args.rgb_input)
    if len(dataset) == 0:
        raise RuntimeError(f"No jpg files found in dataset directory: {args.dataset_dir}")

    num_test = int(args.test_percent * len(dataset))
    if len(dataset) > 1:
        num_test = max(1, min(num_test, len(dataset) - 1))
    else:
        num_test = 0
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 2)
    device = resolve_device(args.device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    Path(args.best_model_path).parent.mkdir(parents=True, exist_ok=True)
    if args.last_model_path:
        Path(args.last_model_path).parent.mkdir(parents=True, exist_ok=True)

    best_loss = 1e9
    print(f"dataset: {len(dataset)} images")
    print(f"train/test: {len(train_dataset)}/{len(test_dataset)}")
    print(f"device: {device}")
    print(f"best model path: {args.best_model_path}")
    print(f"last model path: {args.last_model_path}")
    print("label mode: direction vector from bottom-center to xy label")

    for epoch in tqdm(range(args.epochs)):
        model.train()
        train_loss = 0.0
        for images, labels in iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = direction_loss(outputs, labels)
            train_loss += float(loss)
            loss.backward()
            optimizer.step()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        test_loss_sum = 0.0
        test_error_sum = 0.0
        test_correct = 0
        test_count = 0
        with torch.no_grad():
            for images, labels in iter(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = direction_loss(outputs, labels)
                errors = angular_errors_deg(outputs, labels)
                batch_size = labels.size(0)
                test_loss_sum += float(loss) * batch_size
                test_error_sum += float(errors.sum())
                test_correct += int((errors <= args.test_acc_threshold_deg).sum())
                test_count += batch_size

        test_loss = test_loss_sum / max(test_count, 1)
        test_acc = 100.0 * test_correct / max(test_count, 1)
        test_error_deg = test_error_sum / max(test_count, 1)

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
