import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import os
import csv


import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torchvision.io import read_image

class SmallestNet(nn.Module):
    def __init__(self):
        super(SmallestNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 16 * 16, 32)
        self.fc2 = nn.Linear(32, 9)  # Assuming there are 9 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(x)
        # print(x.shape) 
        x = x.view(-1, 6 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DeeperNet(nn.Module):
    def __init__(self):
        super(DeeperNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 9)  # Assuming there are 2 classes - cats and dogs

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SmallerNet(nn.Module):
    def __init__(self):
        super(SmallerNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 9)  # Assuming there are 9 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(x)
        # print(x.shape) 
        x = x.view(-1, 8 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the saved model
model = SmallestNet()
file_path = "/Users/cglin/Desktop/CNN/SmallestNet/SmallestNet.pth"
model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path

# Use the custom dataset
test_path = "/Users/cglin/Desktop/fig/test"
test_data = CustomImageDataset(test_path, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

id_class_dict = {}
# Use the model to predict the class of each image in the test set
for images, paths in test_loader:
    images = Variable(images)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    # Extract ID from the image path
    id = os.path.basename(paths[0]).split('.')[0]

    # Store ID and corresponding predicted class in the dictionary
    id_class_dict[id] = predicted.item()

for id, class_name in id_class_dict.items():
    print(f"ID: {id}, Class: {class_name}")

# Sort the dictionary by ID in descending order
sorted_dict = dict(sorted(id_class_dict.items(), key=lambda item: int(item[0]), reverse=False))

# Write the sorted dictionary to a CSV file
with open('/Users/cglin/Desktop/CNN/SmallestNet/result.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "label"])  # Write header
    for id, label in sorted_dict.items():
        writer.writerow([id, label+1])  # Write data rows