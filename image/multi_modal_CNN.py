import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import wandb
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")

parser.add_argument("--name", type=str, default="CNN",
                        help="model name")

args = parser.parse_args()

# # 初始化wanda
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="Image-diagnosis",
#     name = args.name,
#     # track hyperparameters and run metadata
#     config={
#     "Learning Rate": args.lr,
#     "Batch size": 64,
#     "model": "CNN"
#     }
# )

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomImageFolder(DatasetFolder):
    def __init__(self, root, csv_file, transform=None, target_transform=None):
        self.csv_data = pd.read_csv(csv_file)
        self.csv_data.set_index('编号', inplace=True)  # assuming 'image_name' is a column in your csv file
        super(CustomImageFolder, self).__init__(root, loader=default_loader, extensions=('jpg', 'jpeg', 'png'), transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        image_name = int(os.path.splitext(os.path.basename(path))[0])
        
        if self.transform is not None:
            sample = self.transform(sample)
        

        # if image_name in self.csv_data.index:
        features = self.csv_data.loc[image_name, ["日产液量", "日产气量"]].values
        features = torch.from_numpy(features.astype('float'))  # Convert features to tensor
        return sample, features, target

        # return sample, features, target


# Define transforms for the training and testing data
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Load the training and testing datasets

train_data = CustomImageFolder('/mnt/mxy/linchungang/image_diagnosis/fig/train', '/mnt/mxy/linchungang/image_diagnosis/dataset/train/train.csv', transform=transform)

# dataset = datasets.ImageFolder(root='/mnt/mxy/linchungang/image_diagnosis/fig/train', transform=transform)

# total_size = len(dataset)
# train_size = int(1 * total_size)  # 100% for training
# val_size = total_size - train_size  # 0% for validation
# train_data, test_data = random_split(dataset, [train_size, val_size])


# test_data = datasets.ImageFolder(root='/mnt/mxy/linchungang/image_diagnosis/fig/test', transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

# class MultiModalNet(nn.Module):
#     def __init__(self):
#         super(MultiModalNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(32 * 8 * 8 + 2, 64)  # Assuming there are 2 extra features
#         self.fc2 = nn.Linear(64, 9)  # Assuming there are 9 classes

#     def forward(self, x, features):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 32 * 8 * 8)
#         print(x.shape)
#         print(features.shape)
#         x = torch.cat((x, features), dim=1)  # Concatenate the CNN features and the extra features
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

class MultiModalNet(nn.Module):
    def __init__(self):
        super(MultiModalNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8194, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 9)  # Assuming there are 2 classes - cats and dogs

    def forward(self, x, features):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        # print(x.shape)
        # print(features.shape)
        
        x = torch.cat((x, features), dim=1)
        
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Evaluate the model
def check_accuracy(test_loader, model):
    correct = 0
    total = 0

    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    print("Accuracy_Test")
    print(accuracy)
    print("Precision_Test")
    print(precision)
    print("Recall_Test")
    print(recall)
    print("F1_Test")
    print(f1)
    # wandb.log({"Accuracy_Test": accuracy})
    # wandb.log({"Precision_Test": precision})
    # wandb.log({"Recall_Test": recall})
    # wandb.log({"F1_Test":  f1})

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


# Instantiate the CNN
model = MultiModalNet()
# model = model.half()
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

report_steps = 30
total_loss = 0
epoch_num = 10
# Train the model
for epoch in range(epoch_num):  # loop over the dataset multiple times

    for i, data in enumerate(train_loader, 0):
        # print(data)
        images, features, labels = data[0].to(device), data[1].to(device), data[2].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images, features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        total_loss += loss.item()
        if (i + 1) % report_steps == 0:
            # logging.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / report_steps))
            print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / report_steps))
            # wandb.log({"loss": total_loss / report_steps})
            total_loss = 0.0
    print("epoch num : ")
    # print(epoch)
    # check_accuracy(test_loader, model)

# print("Final Test")
# check_accuracy(test_loader, model)

torch.save(model.state_dict(), '/mnt/mxy/linchungang/image_diagnosis/full_train/float16/multimodal_CNN.pth')

print('Finished Training')


