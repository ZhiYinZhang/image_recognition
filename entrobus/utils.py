import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader

from time import time
import os
import os.path
import copy
import numpy as np


# define model train

def train_model(model_ft, dataloader, criterion, optimizer, epochs, device=torch.device('cpu'), is_inception=False):
    since = time()
    best_param_dict = copy.deepcopy(model_ft.state_dict())
    best_acc = 0.0
    hist_acc = []
    top1_accuracy = 0.0
    top5_accuracy = 0.0
    best_top5 = 0.0
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 20)
        for x in ['train', 'val']:
            running_losse = 0.0
            model_acc = 0.0
            top5_accuracy = 0.0

            if x not in dataloader:
                break

            with torch.set_grad_enabled(x == 'train'):
                if x == 'train':
                    model_ft.train()
                else:
                    model_ft.eval()

                for inputs, labels in dataloader[x]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    if is_inception and x == 'train':
                        out, aux_out = model_ft(inputs)
                        loss1 = criterion(out, labels)
                        loss2 = criterion(aux_out, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        out = model_ft(inputs)
                        loss = criterion(out, labels)
                    if x == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    _, pred = torch.max(out, 1)
                    running_losse += loss.item() * inputs.size(0)
                    model_acc += torch.sum(pred == labels)
                    out = out.detach()
                    with torch.no_grad():
                        out_5 = np.argsort(out, axis=1)
                        out_5 = out_5[:, -5:]
                        labels = labels.view(-1, 1)
                        labels = labels.data.cpu()
                        top5_accuracy += torch.sum(torch.sum(out_5 == labels, dim=1))
            running_losse = running_losse / len(dataloader[x].dataset)
            model_acc = model_acc.item() / len(dataloader[x].dataset)
            top5_accuracy = top5_accuracy.item() / len(dataloader[x].dataset)
            if x == 'val':
                hist_acc.append(model_acc)
            if model_acc > best_acc:
                best_param_dict = copy.deepcopy(model_ft.state_dict())
                best_acc = model_acc
            if top5_accuracy > best_top5:
                best_top5 = top5_accuracy
            print('{} loss: losses {:.4f} accuracy {:.1f}%'.format(x, running_losse, model_acc*100))
        print()
    print("Training finished. Total cost time: {}min".format((time() - since) // 60))
    if 'val' in dataloader:
        model_ft.load_state_dict(best_param_dict)
    top1_accuracy = best_acc

    return model_ft, hist_acc, top1_accuracy, best_top5


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for parameter in model.parameters():
            parameter.requires_grad = False


# 初始化模型
def initialize_model(model_name, num_class, extract_feature, pretrain=True):
    model_fc = None
    input_size = 0
    if model_name == 'resnet':
        model_fc = models.resnet18(pretrained=pretrain)
        set_parameter_requires_grad(model_fc, extract_feature)
        final_in_features = model_fc.fc.in_features
        model_fc.fc = nn.Linear(final_in_features, num_class)
        input_size = 224
    elif model_name == 'squeezenet':
        model_fc = models.squeezenet1_0(pretrained=pretrain)
        set_parameter_requires_grad(model_fc, extract_feature)
        model_fc.classifier[1] = nn.Conv2d(512, num_class, kernel_size=(1, 1), stride=(1, 1))
        model_fc.num_classes = num_class
        input_size = 224
    elif model_name == 'vgg':
        model_fc = models.vgg11_bn(pretrained=pretrain)
        set_parameter_requires_grad(model_fc, extract_feature)
        final_in_features = model_fc.classifier[6].in_features
        model_fc.classifier[6] = nn.Linear(final_in_features, num_class)
        input_size = 224
    elif model_name == 'alexnet':
        model_fc = models.alexnet(pretrained=pretrain)
        set_parameter_requires_grad(model_fc, extract_feature)
        final_in_features = model_fc.classifier[6].in_features
        model_fc.classifier[6] = nn.Linear(final_in_features, num_class)
        input_size = 224
    elif model_name == 'densnet':
        model_fc = models.densenet121(pretrained=pretrain)
        set_parameter_requires_grad(model_fc, extract_feature)
        final_in_features = model_fc.classifier.in_features
        model_fc.classifier = nn.Linear(final_in_features, num_class)
        input_size = 224
    elif model_name == 'inception':
        model_fc = models.inception_v3(pretrained=pretrain)
        set_parameter_requires_grad(model_fc, extract_feature)
        auxiliary_in_features = model_fc.AuxLogits.fc.in_features
        model_fc.AuxLogits.fc = nn.Linear(auxiliary_in_features, num_class)
        final_in_features = model_fc.fc.in_features
        model_fc.fc = nn.Linear(final_in_features, num_class)
        input_size = 299
    else:
        print("Invalid model name, exiting...")
        exit()
    return model_fc, input_size
