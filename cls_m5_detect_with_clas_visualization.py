from models import *
from utils.utils import *
from utils.datasets import *

import os
import cv2
import torch
import argparse
from PIL import Image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from efficientnet_pytorch import EfficientNet
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


class MicroNucleuClassification(Dataset):
    # def __init__(self, root='/home/jwxie/Desktop/negative_patch_generated', num_classes=2):
    def __init__(self, root='/home/jwxie/Desktop/Micro-Nucleu/PyTorch-YOLOv3/data/classification_patches_old/negative', num_classes=2):
        super(MicroNucleuClassification, self).__init__()

        self.root = root
        self.num_classes = num_classes
        
        self.data_list = [os.path.join(root, e) for e in os.listdir(root)]
        self.label_list = [0 for _ in range(len(self.data_list))]

    def __getitem__(self, item):
        patch = np.array(Image.open(self.data_list[item]))
        label = self.label_list[item]

        return np.transpose(patch, (2, 0, 1)), label, self.data_list[item]

    def __len__(self):
        return len(self.data_list)


def eval_classification(cls_batch_size=64):
    net_type = 2
    epoch = 42
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = EfficientNet.from_pretrained(f'efficientnet-b{net_type}', num_classes=2).to(device)
    model.load_state_dict(torch.load(f'./checkpoints/PickData_Cls_EffNet-b{net_type}-{epoch}.pkl'))
    model.to(device)
    model.eval()

    eval_dataset = MicroNucleuClassification()
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=cls_batch_size, shuffle=False)

    acc, eval_pred, eval_label = [], [], []
    for i, (img, lb, _) in enumerate(eval_loader):
        eval_label.append(lb.cpu().numpy())
        images = img.type(torch.FloatTensor).to(device)
        eval_pred.append(torch.argmax(torch.softmax(model(images), dim=1), dim=1).cpu().detach().numpy())

    tp, tn, fp, fn = 0, 0, 0, 0
    for p, l in zip(eval_pred, eval_label):
        tp += ((p == l) & (l == 1)).sum()
        tn += ((p == l) & (l == 0)).sum()
        fp += ((p == 1) & (l == 0)).sum()
        fn += ((p == 0) & (l == 1)).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    f1 = 2 * precision * recall / (precision + recall)
    print(f'  Evaluation - \n'
          f'  Accuracy: {accuracy},\n'
          f'  Precision: {precision},\n'
          f'  Recall: {recall}, \n'
          f'  F1-score: {f1}')
    print(f'  Confusion Matrix: \n'
          f'            Positive   Negative \n'
          f'  Positive: {tp :^8}   {fp :^8}\n'
          f'  Negative: {fn :^8}   {tn :^8}')


if __name__ == '__main__':
    eval_classification()
