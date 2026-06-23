#!/usr/bin/env python3

# Local copy of train_LaneFollowing.ipynb.
# Changed only the environment-facing parts:
# - no Google Drive copy/unzip cells
# - dataset path can be passed by argument
# - device can be selected by argument
# - output path can be passed by argument

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm


IMAGE_SIZE = 224.0
CAR_X = IMAGE_SIZE / 2.0
CAR_Y = IMAGE_SIZE


def get_label_x(path):
    """Gets the original label x pixel from the image filename."""
    return float(int(path[3:6]))


def get_label_y(path):
    """Gets the original label y pixel from the image filename."""
    return float(int(path[7:10]))


def label_to_direction(label_x, label_y):
    # 차량 현재 위치를 이미지 맨 아래 중앙으로 보고, 라벨 점까지의 선 방향을 정답으로 쓴다.
    dx = float(label_x) - CAR_X
    dy = CAR_Y - float(label_y)
    norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    return dx / norm, dy / norm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def normalize_vectors(vectors):
    return vectors / torch.clamp(torch.norm(vectors, dim=1, keepdim=True), min=1e-6)


def direction_loss(outputs, labels):
    # 모델 출력 크기보다 방향이 중요하므로 출력 벡터를 단위벡터로 정규화한 뒤 비교한다.
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
            # Same as the original notebook. This matches OpenCV BGR camera frames.
            image = image.numpy()[::-1].copy()
            image = torch.from_numpy(image)
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return image, torch.tensor([vx, vy]).float()


def build_parser():
    parser = argparse.ArgumentParser(description="Local direction-vector copy of train_LaneFollowing.ipynb")
    parser.add_argument("--dataset-dir", default="LaneFollowing/dataset_xy")
    parser.add_argument("--best-model-path", default="LaneFollowing/best_model_direction.pth")
    parser.add_argument("--last-model-path", default="LaneFollowing/last_model_direction.pth")
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

    test_percent = args.test_percent
    num_test = int(test_percent * len(dataset))
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

    NUM_EPOCHS = args.epochs
    BEST_MODEL_PATH = args.best_model_path
    LAST_MODEL_PATH = args.last_model_path
    Path(BEST_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    if LAST_MODEL_PATH:
        Path(LAST_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    best_loss = 1e9

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1)
    epoch_list = []
    train_loss_list = []
    test_loss_list = []

    print(f"dataset: {len(dataset)} images")
    print(f"train/test: {len(train_dataset)}/{len(test_dataset)}")
    print(f"device: {device}")
    print(f"best model path: {BEST_MODEL_PATH}")
    print(f"last model path: {LAST_MODEL_PATH}")
    print(f"label mode: direction vector from bottom-center to xy label")
    print(f"test acc threshold: {args.test_acc_threshold_deg:.1f}deg")

    for epoch in tqdm(range(NUM_EPOCHS)):
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
            # scheduler.step()
        train_loss /= len(train_loader)

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

        print(
            "train loss %f, test_loss %f, test_acc %.2f%%, test_err_deg %.2f, lr %f"
            % (train_loss, test_loss, test_acc, test_error_deg, (get_lr(optimizer)))
        )

        epoch_list.append(epoch)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        if LAST_MODEL_PATH:
            torch.save(model.state_dict(), LAST_MODEL_PATH)
        if test_loss < best_loss:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_loss = test_loss
            print(f"saved best model: {BEST_MODEL_PATH} test_loss={best_loss:.6f}")


if __name__ == "__main__":
    main()
