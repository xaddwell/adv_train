"""
Evaluation with AutoAttack.
"""

import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
from collections import OrderedDict
from core.models import create_model
from torchvision import transforms as T
from core.utils import Logger
from torch.utils.data import DataLoader
from torchvision import datasets
import torchattacks as ta


DATA_DIR = CIFAR10_DATA_DIR = r"D:\cjh\Adversarial_Robustness\datasets\CIFAR10"
WEIGHTS = r"D:\cjh\Adversarial_Robustness\third_party\DM-Improves-AT\ckpt\cifar10_l2_wrn28-10.pt"


info = {
    'data': 'cifar10',
    'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    'num_classes': 10,
    'mean': [0.4914, 0.4822, 0.4465],
    'std': [0.2023, 0.1994, 0.2010],
}
# BATCH_SIZE = args.batch_size
# BATCH_SIZE_VALIDATION = args.batch_size_validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load data
model_name = "wrn-28-10"
batch_size = 64
samples_num = 64
indices = range(samples_num)

to_tensor = T.ToTensor()
normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
postprocess = T.Compose([to_tensor])

dataset = datasets.CIFAR10(root=CIFAR10_DATA_DIR,train=False,transform=postprocess)
dataset = torch.utils.data.Subset(dataset, indices)
loader = DataLoader(dataset, batch_size=batch_size)


# Model

model = create_model(model_name, normalize = False, info = info, device = device)
checkpoint = torch.load(WEIGHTS)
new_state_dict = OrderedDict()
for key, value in checkpoint.items():
    name = 'module.' + key
    new_state_dict[name] = value
model.load_state_dict(new_state_dict)

model.eval()
del checkpoint


def get_attack(_method, target_model):
    # target没有做
    if _method == "PGD":
        atk = ta.PGD(target_model, eps= 8 / 255, alpha=1 / 255, steps=40)
    elif _method == "PGDL2":
        atk = ta.PGDL2(target_model, eps= 1, alpha=0.1, steps=40)
    else:
        atk = None

    return atk


# Evaluation.
if __name__ == '__main__':
    PGD = get_attack("PGD", model)
    PGDL2 = get_attack("PGDL2", model)
    correct = 0;adv = 0;adv_l2 = 0

    for images, labels in tqdm.tqdm(loader):
        images = images.cuda()
        labels = labels.cuda()
        pred = model(images).argmax(dim=1)
        adv += int((model(PGD(images, labels)).argmax(dim=1) == labels).sum())
        adv_l2 += int((model(PGDL2(images, labels)).argmax(dim=1) == labels).sum())
        correct += int((pred == labels).sum())

    print("Method:{} Clean:{:.3f} PGD:{:.3f} PGDL2:{:.3f}".format(
      model_name, correct / samples_num, adv / samples_num, adv_l2 / samples_num
    ))