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

feature_list = ["日产液量", "日产气量","displacement_mean", "displacement_median", "displacement_std", "displacement_var", "displacement_kurtosis", "displacement_skew", "payload_mean", "payload_median", "payload_std", "payload_var", "payload_kurtosis", "payload_skew", "displacementblock0_mean", "displacementblock0_median", "displacementblock0_standard_deviation", "displacementblock0_variance", "displacementblock0_kurtosis_value", "displacementblock0_skew_value", "displacementblock1_mean", "displacementblock1_median", "displacementblock1_standard_deviation", "displacementblock1_variance", "displacementblock1_kurtosis_value", "displacementblock1_skew_value", "displacementblock2_mean", "displacementblock2_median", "displacementblock2_standard_deviation", "displacementblock2_variance", "displacementblock2_kurtosis_value", "displacementblock2_skew_value", "displacementblock3_mean", "displacementblock3_median", "displacementblock3_standard_deviation", "displacementblock3_variance", "displacementblock3_kurtosis_value", "displacementblock3_skew_value", "displacementblock4_mean", "displacementblock4_median", "displacementblock4_standard_deviation", "displacementblock4_variance", "displacementblock4_kurtosis_value", "displacementblock4_skew_value", "displacementblock5_mean", "displacementblock5_median", "displacementblock5_standard_deviation", "displacementblock5_variance", "displacementblock5_kurtosis_value", "displacementblock5_skew_value", "displacementblock6_mean", "displacementblock6_median", "displacementblock6_standard_deviation", "displacementblock6_variance", "displacementblock6_kurtosis_value", "displacementblock6_skew_value", "displacementblock7_mean", "displacementblock7_median", "displacementblock7_standard_deviation", "displacementblock7_variance", "displacementblock7_kurtosis_value", "displacementblock7_skew_value", "displacementblock8_mean", "displacementblock8_median", "displacementblock8_standard_deviation", "displacementblock8_variance", "displacementblock8_kurtosis_value", "displacementblock8_skew_value", "displacementblock9_mean", "displacementblock9_median", "displacementblock9_standard_deviation", "displacementblock9_variance", "displacementblock9_kurtosis_value", "displacementblock9_skew_value", "displacementblock10_mean", "displacementblock10_median", "displacementblock10_standard_deviation", "displacementblock10_variance", "displacementblock10_kurtosis_value", "displacementblock10_skew_value", "displacementblock11_mean", "displacementblock11_median", "displacementblock11_standard_deviation", "displacementblock11_variance", "displacementblock11_kurtosis_value", "displacementblock11_skew_value", "payloadblock0_mean", "payloadblock0_median", "payloadblock0_standard_deviation", "payloadblock0_variance", "payloadblock0_kurtosis_value", "payloadblock0_skew_value", "payloadblock1_mean", "payloadblock1_median", "payloadblock1_standard_deviation", "payloadblock1_variance", "payloadblock1_kurtosis_value", "payloadblock1_skew_value", "payloadblock2_mean", "payloadblock2_median", "payloadblock2_standard_deviation", "payloadblock2_variance", "payloadblock2_kurtosis_value", "payloadblock2_skew_value", "payloadblock3_mean", "payloadblock3_median", "payloadblock3_standard_deviation", "payloadblock3_variance", "payloadblock3_kurtosis_value", "payloadblock3_skew_value", "payloadblock4_mean", "payloadblock4_median", "payloadblock4_standard_deviation", "payloadblock4_variance", "payloadblock4_kurtosis_value", "payloadblock4_skew_value", "payloadblock5_mean", "payloadblock5_median", "payloadblock5_standard_deviation", "payloadblock5_variance", "payloadblock5_kurtosis_value", "payloadblock5_skew_value", "payloadblock6_mean", "payloadblock6_median", "payloadblock6_standard_deviation", "payloadblock6_variance", "payloadblock6_kurtosis_value", "payloadblock6_skew_value", "payloadblock7_mean", "payloadblock7_median", "payloadblock7_standard_deviation", "payloadblock7_variance", "payloadblock7_kurtosis_value", "payloadblock7_skew_value", "payloadblock8_mean", "payloadblock8_median", "payloadblock8_standard_deviation", "payloadblock8_variance", "payloadblock8_kurtosis_value", "payloadblock8_skew_value", "payloadblock9_mean", "payloadblock9_median", "payloadblock9_standard_deviation", "payloadblock9_variance", "payloadblock9_kurtosis_value", "payloadblock9_skew_value", "payloadblock10_mean", "payloadblock10_median", "payloadblock10_standard_deviation", "payloadblock10_variance", "payloadblock10_kurtosis_value", "payloadblock10_skew_value", "payloadblock11_mean", "payloadblock11_median", "payloadblock11_standard_deviation", "payloadblock11_variance", "payloadblock11_kurtosis_value", "payloadblock11_skew_value",]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")

parser.add_argument("--name", type=str, default="CNN",
                        help="model name")

args = parser.parse_args()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomImageFolder(DatasetFolder):
    def __init__(self, root, csv_file, transform=None, target_transform=None):
        self.csv_data = pd.read_csv(csv_file)
        self.csv_data.set_index('编号', inplace=True)  # assuming 'image_name' is a column in your csv file
        
        # 将第4列和第5列的NaN值替换为各自列的平均值
        self.csv_data['日产液量'].fillna(-1, inplace=True)
        self.csv_data['日产气量'].fillna(-1, inplace=True)
        for feature in feature_list:
            self.csv_data[feature].fillna(-1, inplace=True)

        super(CustomImageFolder, self).__init__(root, loader=default_loader, extensions=('jpg', 'jpeg', 'png'), transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        image_name = int(os.path.splitext(os.path.basename(path))[0])
        
        if self.transform is not None:
            sample = self.transform(sample)
        

        # if image_name in self.csv_data.index:
        # print(self.csv_data.columns)
        features = self.csv_data.loc[image_name, feature_list].values
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

dataset = CustomImageFolder('/mnt/mxy/linchungang/image_diagnosis/fig/train', '/mnt/mxy/linchungang/image_diagnosis/dataset/train_add2.csv', transform=transform)

# dataset = datasets.ImageFolder(root='/mnt/mxy/linchungang/image_diagnosis/fig/train', transform=transform)

total_size = len(dataset)
train_size = int(0.9 * total_size)  # 100% for training
val_size = total_size - train_size  # 0% for validation
train_data, test_data = random_split(dataset, [train_size, val_size])


# test_data = datasets.ImageFolder(root='/mnt/mxy/linchungang/image_diagnosis/fig/test', transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

class Attention(nn.Module):
    def __init__(self, in_features, out_features):
        super(Attention, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        scores = self.fc(x)
        scores = self.softmax(scores)
        x = x * scores
        return x

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

# Evaluate the model
def check_accuracy(test_loader, model):
    correct = 0
    total = 0

    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for data in test_loader:
            images, features, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            outputs = model(images,features)
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
model = MultiModalNet_A()
# model = model.half()
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)


report_steps = 30
total_loss = 0
epoch_num = 20
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
    print(epoch)
    check_accuracy(test_loader, model)

# print("Final Test")
# check_accuracy(test_loader, model)

torch.save(model.state_dict(), '/mnt/mxy/linchungang/image_diagnosis/full_train/float32/MultiModalNet_F.pth')

print('Finished Training')


