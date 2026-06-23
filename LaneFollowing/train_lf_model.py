import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import glob
import PIL.Image
import os
import numpy as np
from tqdm import tqdm

center = 224./2.

def get_x(path):
    """Gets the x value from the image filename"""
    #return float(int(path[3:6]))
    return (float(int(path[3:6]))/100.) - (center/100.)

def get_y(path):
    """Gets the y value from the image filename"""
    #return float(int(path[7:10]))
    return (float(int(path[7:10])) - 50.0) / 50.0


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class XYDataset(torch.utils.data.Dataset):

    def __init__(self, directory, random_hflips=False):
        self.directory = directory
        self.random_hflips = random_hflips
        self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = PIL.Image.open(image_path)
        x = float(get_x(os.path.basename(image_path)))
        y = float(get_y(os.path.basename(image_path)))

        if float(np.random.rand(1)) > 0.5:
            image = transforms.functional.hflip(image)
            x = -x

        image = self.color_jitter(image)
        image = transforms.functional.to_tensor(image)
        image = image.numpy()[::-1].copy()
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return image, torch.tensor([x, y]).float()

dataset = XYDataset('dataset_xy', random_hflips=False)


test_percent = 0.2
num_test = int(test_percent * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)


model = models.resnet18(pretrained=True)


model.fc = torch.nn.Linear(512, 2)
device = torch.device('cuda')
model = model.to(device)

NUM_EPOCHS = 81
BEST_MODEL_PATH = 'best_model_xy.pth'
best_loss = 1e9

optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
#scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1)
epoch_list = []
train_loss_list = []
test_loss_list = []

for epoch in tqdm(range(NUM_EPOCHS)):

    model.train()
    train_loss = 0.0
    for images, labels in iter(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.mse_loss(outputs, labels)
        train_loss += float(loss)
        loss.backward()
        optimizer.step()
        #scheduler.step() 
    train_loss /= len(train_loader)

    model.eval()
    test_loss = 0.0
    for images, labels in iter(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = F.mse_loss(outputs, labels)
        test_loss += float(loss)
    test_loss /= len(test_loader)

    print('train loss %f, test_loss %f, lr %f' % (train_loss, test_loss, (get_lr(optimizer))))

    epoch_list.append(epoch)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    if test_loss < best_loss:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_loss = test_loss


import matplotlib.pyplot as plt

plt.plot(epoch_list, train_loss_list, '-b', label='train_loss_list')
plt.plot(epoch_list, test_loss_list, '-r', label='test_loss')

plt.xlabel("n iteration")
plt.legend(loc='upper left')

# show
plt.show()        
