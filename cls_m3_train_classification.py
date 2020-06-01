import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import linecache
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
from torch.optim import Adam
from PIL import Image
from torchvision.models import vgg16_bn
from tensorboardX import SummaryWriter

# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()


class MicroNucleuClassificationTxt(Dataset):
    def __init__(self, root='./', num_classes=2, mode='train'):
        super(MicroNucleuClassificationTxt, self).__init__()

        self.root = root
        self.num_classes = num_classes
        self.data_list = linecache.getlines(os.path.join(root, f'{mode}_list.txt'))

        random.shuffle(self.data_list)

    def __getitem__(self, item):
        line = self.data_list[item].rstrip()
        name, label = line.split(' ')
        patch = np.array(Image.open(name))

        return np.transpose(patch, (2, 0, 1)), int(label), name.split('/')[-1]

    def __len__(self):
        return len(self.data_list)


def train_classification(lr=0.001, epochs=200, cls_batch_size=64):
    net_type = 2
    model = EfficientNet.from_pretrained(f'efficientnet-b{net_type}', num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(np.array([0.74, 1.25])).type(torch.FloatTensor)).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):

        # prepare data
        train_dataset = MicroNucleuClassificationTxt()
        train_loader = DataLoader(dataset=train_dataset, batch_size=cls_batch_size, shuffle=True)

        eval_dataset = MicroNucleuClassificationTxt(mode='validation')
        eval_loader = DataLoader(dataset=eval_dataset, batch_size=cls_batch_size, shuffle=True)
        total_step = len(train_loader)

        # set model
        model.train()
        forward_loss = []
        for i, (img, lb, _) in enumerate(train_loader):
            images = img.type(torch.FloatTensor).to(device)
            labels = lb.type(torch.LongTensor).to(device)

            output = model(images)
            loss = criterion(output, labels)
            forward_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('train_loss', np.mean(forward_loss), epoch * total_step + i)
            print(f'Train - Epoch: [{epoch + 1}/{epochs}], step: [{i + 1}/{total_step}], loss: {np.mean(forward_loss)}')

        model.eval()
        acc, eval_pred, eval_label = [], [], []
        for i, (img, lb, _) in enumerate(eval_loader):
            eval_label.append(lb.cpu().numpy())
            images = img.type(torch.FloatTensor).to(device)
            eval_pred.append(torch.argmax(model(images), dim=1).cpu().detach().numpy())

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
        print(f'  Evaluation - '
              f'  Accuracy: {accuracy},\n'
              f'  Precision: {precision},\n'
              f'  Recall: {recall}, \n'
              f'  F1-score: {f1}')
        print(f'  Confusion Matrix: \n'
              f'            Positive   Negative \n'
              f'  Positive: {tp :^8}   {fp :^8}\n'
              f'  Negative: {fn :^8}   {tn :^8}')
        writer.add_scalar('test_accuracy', accuracy, epoch)
        writer.add_scalar('test_F1', f1, epoch)
        writer.add_scalar('test_precision', precision, epoch)
        writer.add_scalar('test_recall', recall, epoch)
        torch.save(model.state_dict(), f'./checkpoints/PickData_Cls_EffNet-b{net_type}-{epoch}.pkl')


if __name__ == '__main__':
    train_classification()
