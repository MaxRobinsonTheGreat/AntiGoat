import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils, datasets, models
from tqdm import tqdm
import cv2
from os import listdir
from model import GoatDetector

class GoatDataset(Dataset):
    def __init__(self, size=64):
        self.size = size
        self.transform = transforms.ToTensor()
        self.images = []

        goat_im_names = listdir("./data/goats/")
        self.num_goats = len(goat_im_names)
        for im_name in goat_im_names:
            # print(im_name)
            self.images.append(self.load_image("./data/goats/"+im_name))
        
        nongoat_im_names = listdir("./data/nongoats/")
        for im_name in nongoat_im_names:
            self.images.append(self.load_image("./data/nongoats/"+im_name))
        
        self.total_ims = len(self.images)

    def load_image(self, path):
        im = cv2.imread(path)[:, :, ::-1]
        im = cv2.resize(im, (self.size, self.size))
        im = self.transform(im)
        return im


    def __getitem__(self, index):
        is_goat = index<self.num_goats
        return self.images[index], torch.tensor(is_goat).float()

    def __len__(self):
        return self.total_ims

print("Init dataset")
dataset = GoatDataset()
batch_size = 16
validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_size = len(train_indices)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_size = len(val_indices)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

print("Init Model")
bne = nn.BCELoss()
model = GoatDetector()
model.cuda()
optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)

accuracies = []
losses = []

print("Training")
epochs = 13
for epoch in range(epochs):

  loop = tqdm(total=len(train_loader), position=0)
  total_loss= 0
  total_accuracy = 0 

  for i, (image, label) in enumerate(train_loader):
    image, label = image.cuda(), label.cuda()
    optimizer.zero_grad()

    out = model(image).squeeze()

    loss = bne(out, label)
    loss.backward()
    optimizer.step()
    total_loss+= loss.item()

    preds = (out >= 0.5).float()
    accuracy = (preds == label).sum().item() / len(label)
    accuracies.append(accuracy)

    total_accuracy += accuracy
    losses.append(loss.item())

    loop.set_description('epoch:{:d} loss:{:.4f} accuracy:{:.2f}'.format(epoch, total_loss/(i+1), total_accuracy/(i+1)))
    loop.update(1)
  loop.close()

  total_correct = 0
  positives = 0
  for i, (image, label) in enumerate(validation_loader):
    image, label = image.cuda(), label.cuda()
    out = model(image)
    preds = (out >= 0.5).float().squeeze(1)
    total_correct += (preds == label).sum().item()
    positives += label.sum().item()
  print(f"Validation accuracy: {total_correct/validation_size:.2f} ({positives/validation_size:.2f} positives)")


print("Accuracy")
plt.plot(accuracies)
plt.show()

print("Loss")
plt.plot(losses)
plt.show()

torch.save(model.state_dict(), "./saved_models/temp")

# import torch
# import gc
# for obj in gc.get_objects():
#     try:
#         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#             print(type(obj), obj.size())
#     except:
#         pass