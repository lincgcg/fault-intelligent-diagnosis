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

feature_list = ["日产液量", "日产气量","displacement_mean", "displacement_median", "displacement_std", "displacement_var", "displacement_kurtosis", "displacement_skew", "payload_mean", "payload_median", "payload_std", "payload_var", "payload_kurtosis", "payload_skew", "displacementblock0_mean", "displacementblock0_median", "displacementblock0_standard_deviation", "displacementblock0_variance", "displacementblock0_kurtosis_value", "displacementblock0_skew_value", "displacementblock1_mean", "displacementblock1_median", "displacementblock1_standard_deviation", "displacementblock1_variance", "displacementblock1_kurtosis_value", "displacementblock1_skew_value", "displacementblock2_mean", "displacementblock2_median", "displacementblock2_standard_deviation", "displacementblock2_variance", "displacementblock2_kurtosis_value", "displacementblock2_skew_value", "displacementblock3_mean", "displacementblock3_median", "displacementblock3_standard_deviation", "displacementblock3_variance", "displacementblock3_kurtosis_value", "displacementblock3_skew_value", "displacementblock4_mean", "displacementblock4_median", "displacementblock4_standard_deviation", "displacementblock4_variance", "displacementblock4_kurtosis_value", "displacementblock4_skew_value", "displacementblock5_mean", "displacementblock5_median", "displacementblock5_standard_deviation", "displacementblock5_variance", "displacementblock5_kurtosis_value", "displacementblock5_skew_value", "displacementblock6_mean", "displacementblock6_median", "displacementblock6_standard_deviation", "displacementblock6_variance", "displacementblock6_kurtosis_value", "displacementblock6_skew_value", "displacementblock7_mean", "displacementblock7_median", "displacementblock7_standard_deviation", "displacementblock7_variance", "displacementblock7_kurtosis_value", "displacementblock7_skew_value", "displacementblock8_mean", "displacementblock8_median", "displacementblock8_standard_deviation", "displacementblock8_variance", "displacementblock8_kurtosis_value", "displacementblock8_skew_value", "displacementblock9_mean", "displacementblock9_median", "displacementblock9_standard_deviation", "displacementblock9_variance", "displacementblock9_kurtosis_value", "displacementblock9_skew_value", "displacementblock10_mean", "displacementblock10_median", "displacementblock10_standard_deviation", "displacementblock10_variance", "displacementblock10_kurtosis_value", "displacementblock10_skew_value", "displacementblock11_mean", "displacementblock11_median", "displacementblock11_standard_deviation", "displacementblock11_variance", "displacementblock11_kurtosis_value", "displacementblock11_skew_value", "payloadblock0_mean", "payloadblock0_median", "payloadblock0_standard_deviation", "payloadblock0_variance", "payloadblock0_kurtosis_value", "payloadblock0_skew_value", "payloadblock1_mean", "payloadblock1_median", "payloadblock1_standard_deviation", "payloadblock1_variance", "payloadblock1_kurtosis_value", "payloadblock1_skew_value", "payloadblock2_mean", "payloadblock2_median", "payloadblock2_standard_deviation", "payloadblock2_variance", "payloadblock2_kurtosis_value", "payloadblock2_skew_value", "payloadblock3_mean", "payloadblock3_median", "payloadblock3_standard_deviation", "payloadblock3_variance", "payloadblock3_kurtosis_value", "payloadblock3_skew_value", "payloadblock4_mean", "payloadblock4_median", "payloadblock4_standard_deviation", "payloadblock4_variance", "payloadblock4_kurtosis_value", "payloadblock4_skew_value", "payloadblock5_mean", "payloadblock5_median", "payloadblock5_standard_deviation", "payloadblock5_variance", "payloadblock5_kurtosis_value", "payloadblock5_skew_value", "payloadblock6_mean", "payloadblock6_median", "payloadblock6_standard_deviation", "payloadblock6_variance", "payloadblock6_kurtosis_value", "payloadblock6_skew_value", "payloadblock7_mean", "payloadblock7_median", "payloadblock7_standard_deviation", "payloadblock7_variance", "payloadblock7_kurtosis_value", "payloadblock7_skew_value", "payloadblock8_mean", "payloadblock8_median", "payloadblock8_standard_deviation", "payloadblock8_variance", "payloadblock8_kurtosis_value", "payloadblock8_skew_value", "payloadblock9_mean", "payloadblock9_median", "payloadblock9_standard_deviation", "payloadblock9_variance", "payloadblock9_kurtosis_value", "payloadblock9_skew_value", "payloadblock10_mean", "payloadblock10_median", "payloadblock10_standard_deviation", "payloadblock10_variance", "payloadblock10_kurtosis_value", "payloadblock10_skew_value", "payloadblock11_mean", "payloadblock11_median", "payloadblock11_standard_deviation", "payloadblock11_variance", "payloadblock11_kurtosis_value", "payloadblock11_skew_value",]

class MultiModalNet_A(nn.Module):
    def __init__(self):
        super(MultiModalNet_A, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc0 = nn.Linear(158, 128 * 8)
        self.fc1 = nn.Linear(128 * 8 * 8 + 128 * 8, 500)
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
model = MultiModalNet_A()
file_path = "/Users/cglin/Desktop/CNN_0715/fulltrain/float32/MultiModalNet_F/MultiModalNet_F.pth"
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
        features = self.csv_data.loc[image_name, feature_list].values
        features = torch.from_numpy(features.astype('float'))
        return sample, features, image_name

# Use the custom dataset
# test_path = "/Users/cglin/Desktop/fig/test"
# test_data = CustomImageDataset(test_path, transform=transform)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

dataset = TestImageFolder('/Users/cglin/Desktop/fig/test', '/Users/cglin/Desktop/test_add2.csv', transform=transform)

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
with open('/Users/cglin/Desktop/CNN_0715/fulltrain/float32/MultiModalNet_F/result.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "label"])  # Write header
    for id, label in sorted_dict.items():
        writer.writerow([id, label+1])  # Write data rows