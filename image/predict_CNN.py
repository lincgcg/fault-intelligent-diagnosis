import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import os
import csv

import torch.nn as nn
import torch.optim as optim
import pandas as pd

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader


class MultiModalNet(nn.Module):
    def __init__(self):
        super(MultiModalNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc0 = nn.Linear(2, 10)
        self.fc1 = nn.Linear(128 * 8 * 8 + 10, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 9)  # Assuming there are 2 classes - cats and dogs

    def forward(self, x, features):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        features = features.to(x.dtype)
        features = F.relu(self.fc0(features))
        x = torch.cat((x, features), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load the saved model
model = MultiModalNet()
file_path = "/Users/cglin/Desktop/CNN_0715/fulltrain/float16/multimodal_CNN/multimodal_CNN.pth"
model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class TestImageFolder(DatasetFolder):
    def __init__(self, root, csv_file, transform=None):
        self.img_dir = root
        self.img_names = os.listdir(root)
        self.transform = transform
        
        self.csv_data = pd.read_csv(csv_file)
        self.csv_data.set_index('编号', inplace=True)
        
        # 将第4列和第5列的NaN值替换为各自列的平均值
        self.csv_data['日产液量'].fillna(-1, inplace=True)
        self.csv_data['日产气量'].fillna(-1, inplace=True)
        # super(TestImageFolder, self).__init__(root, loader=default_loader, extensions=('jpg', 'jpeg', 'png'), transform=transform)
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_names[index])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # path, _ = self.sample[index]
        # sample = self.loader(path)
        sample = image
        image_name = int(os.path.splitext(os.path.basename(img_path))[0])  # remove extension from filename

        # if self.transform is not None:
        #     sample = self.transform(sample)

        # if image_name in self.csv_data.index:
        features = self.csv_data.loc[image_name, ["日产液量", "日产气量"]].values
        features = torch.from_numpy(features.astype('float'))
        return sample, features, image_name

# Use the custom dataset
# test_path = "/Users/cglin/Desktop/fig/test"
# test_data = CustomImageDataset(test_path, transform=transform)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

dataset = TestImageFolder('/Users/cglin/Desktop/fig/test', '/Users/cglin/Desktop/0715比赛/dataset/test/test.csv', transform=transform)

test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

id_class_dict = {}
# Use the model to predict the class of each image in the test set
for sample, features, image_name in test_loader:
    outputs = model(sample, features)
    _, predicted = torch.max(outputs.data, 1)

    # Extract ID from the image path
    id = int(image_name)

    # Store ID and corresponding predicted class in the dictionary
    id_class_dict[id] = predicted.item()

for id, class_name in id_class_dict.items():
    print(f"ID: {id}, Class: {class_name}")

# Sort the dictionary by ID in descending order
sorted_dict = dict(sorted(id_class_dict.items(), key=lambda item: int(item[0]), reverse=False))

# Write the sorted dictionary to a CSV file
with open('/Users/cglin/Desktop/CNN_0715/fulltrain/float16/multimodal_CNN/result.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "label"])  # Write header
    for id, label in sorted_dict.items():
        writer.writerow([id, label+1])  # Write data rows