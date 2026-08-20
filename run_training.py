import os
os.environ['CURL_CA_BUNDLE'] = ''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np
import time
from tqdm import tqdm
import ssl

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

import urllib.request
urllib.request._create_unverified_context = ssl._create_unverified_context

exec(open('resnet9_starter.py').read().split('if __name__')[0])

if __name__ == '__main__':
    main()
