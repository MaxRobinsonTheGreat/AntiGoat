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
from PIL import Image
from os import listdir
from model import GoatDetector
import random
import time

def calc_pos_neg(pred, actual):
    ones = torch.ones(pred.size()).cuda()
    zeros = torch.zeros(pred.size()).cuda()
    true_pos = torch.sum(torch.where((pred == 1) & (actual == 1), ones, zeros))
    false_pos = torch.sum(torch.where((pred == 1) & (actual == 0), ones, zeros))
    true_neg = torch.sum(torch.where((pred == 0) & (actual == 0), ones, zeros))
    false_neg = torch.sum(torch.where((pred == 0) & (actual == 1), ones, zeros))
    tot_pos = true_pos + false_neg
    tot_neg = true_neg + false_pos
    return true_pos, false_pos, true_neg, false_neg, tot_pos, tot_neg

def calc_accuracy(model, loader, threshold=0.5):
    total_correct = 0
    positives = 0
    count = 0
    for i, (image, label) in enumerate(loader):
        image, label = image.cuda(), label.cuda()
        count += len(image)
        out = model(image)
        preds = (out >= threshold).float().squeeze(1)
        total_correct += (preds == label).sum().item()

        positives += label.sum().item()
    return total_correct/count, positives/count

def threshold_search(model, loader, num_thresh=10):
    interval = 1/num_thresh
    highest_acc = 0
    best_thresh = 0
    thresholds = []
    accuracies = []
    for i in tqdm(range(num_thresh)):
        threshold = i*interval
        acc, _ = calc_accuracy(model, loader, threshold=threshold)
        if acc > highest_acc:
            best_thresh = threshold
            highest_acc = acc
        thresholds.append(threshold)
        accuracies.append(acc)
    return best_thresh, highest_acc, thresholds, accuracies

# def calc_val_accuracy()


# p=torch.tensor([1, 1, 1 , 1, 1, 1, 1, 1, 1])
# a=torch.tensor([0, 0, 1 , 1, 1, 1, 0, 0, 0])
# print(calc_pos_neg(p, a))
# exit()
class GoatDataset(Dataset):
    def __init__(self, size=64, train=True):
        self.train=train
        self.size = size
        self.transform = transforms.ToTensor()
        self.images = []
        self.totensor = transforms.ToTensor()

        goat_im_names = listdir("./data/goats/")
        for im_name in goat_im_names:
            self.images.append(self.load_image("./data/goats/"+im_name))
            # self.images.append(self.load_image("./data/goats/"+im_name, True))
        self.num_goats = len(self.images)
        
        nongoat_im_names = listdir("./data/nongoats/")
        for im_name in nongoat_im_names:
            self.images.append(self.load_image("./data/nongoats/"+im_name))
            # self.images.append(self.load_image("./data/nongoats/"+im_name, True))
        
        self.total_ims = len(self.images)
        percent_goats = self.num_goats/self.total_ims
        percent_nongoats = (self.total_ims-self.num_goats)/self.total_ims

        print("Percent goats:", percent_goats, "Percent nongoats:", percent_nongoats)

    def load_image(self, path, use_preprocessing=False):
        im = Image.open(path)
        if use_preprocessing:
            preprocessing = transforms.Compose(
                            [
                                transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
                                transforms.RandomHorizontalFlip()
                                # transforms.RandomCrop(64-random.randint(0, 20)),
                                # transforms.Resize(64)
                            ])
            im = preprocessing(im)

        # im = cv2.imread(path)[:, :, ::-1]
        # im = cv2.resize(im, (self.size, self.size))
        # im = self.transform(im)
        return im


    def __getitem__(self, index):
        is_goat = index<self.num_goats
        image = self.images[index]
        # if self.train:
            # if is_goat:
            #     preprocessing = transforms.Compose(
            #             [#transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
            #             transforms.RandomHorizontalFlip(0.5)])
            # # if is_goat:
            # #     preprocessing = transforms.Compose(
            # #         [transforms.ColorJitter(0.5, 0.5, 0.5, 0.2)])
            # # else:
            # #     preprocessing = transforms.Compose(
            # #         [transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
            # #         transforms.RandomCrop(64-random.randint(0, 20)),
            # #         transforms.Resize(64)])
            #     image = preprocessing(image)

            
        image = dataset.totensor(image)
        return image, torch.tensor(is_goat).float()

    def __len__(self):
        return self.total_ims

print("Init dataset")
dataset = GoatDataset()
val_dataset = GoatDataset(train=False)
batch_size = 16
validation_split = .2
shuffle_dataset = True

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    # np.random.seed()
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_size = len(train_indices)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_size = len(val_indices)
validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

print("Init Model")
bne = nn.BCELoss()
model = GoatDetector()
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params,weight_decay=1e-5)

accuracies = []
losses = []

print("Training")
epochs = 10
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
    val_acc, breakdown = calc_accuracy(model, validation_loader)
    print(f"Validation accuracy:{val_acc:.2f} ({breakdown:.2f} pos)")


print("Threshold searching...")
best_thresh, highest_acc, thresholds, thresh_accuracies = threshold_search(model, validation_loader, 25)

print(f"Best Threshold:{best_thresh:.2f} Accuracy: {highest_acc:.2f}")
plt.plot(thresholds, thresh_accuracies)
plt.show()

print("Training Accuracy")
plt.plot(accuracies)
plt.show()

print("Training Loss")
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