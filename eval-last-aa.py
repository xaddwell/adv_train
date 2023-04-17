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

import torch
import torch.nn as nn

from autoattack import AutoAttack
    
from core.data import get_data_info
from core.data import load_data
from core.models import create_model

from core.utils import Logger
from core.utils import parser_eval
from core.utils import seed
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets


# Setup

parse = parser_eval()
args = parse.parse_args()

LOG_DIR = args.log_dir + '/' + args.desc

DATA_DIR = CIFAR10_DATA_DIR = r"D:\cjh\Adversarial_Robustness\datasets\CIFAR10"
WEIGHTS = r"D:\cjh\Adversarial_Robustness\third_party\DM-Improves-AT\ckpt\cifar10_linf_wrn28-10.pt"

info = {
    'data': 'cifar10',
    'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    'num_classes': 10,
    'mean': [0.4914, 0.4822, 0.4465],
    'std': [0.2023, 0.1994, 0.2010],
}
# BATCH_SIZE = args.batch_size
# BATCH_SIZE_VALIDATION = args.batch_size_validation
BATCH_SIZE = 128
BATCH_SIZE_VALIDATION = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load data

seed(args.seed)
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



l = [x for (x, y) in loader]
x_test = torch.cat(l, 0)
l = [y for (x, y) in loader]
y_test = torch.cat(l, 0)

args.model,args.normalize = "wrn-28-10",True

# Model
print(args.model)
model = create_model(args.model, args.normalize, info, device)
checkpoint = torch.load(WEIGHTS)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
del checkpoint

