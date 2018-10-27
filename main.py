#!/usr/bin/env python

# --------------------------------------------------------
# Image Training and Reconition
# Copyright (c) 2018 Entrobus
# Licensed under The MIT License [see LICENSE for details]
# Written by Genieliu
# --------------------------------------------------------

"""Train and test the Image Reconiton Task with the data provided by user"""

import argparse
import numpy as np
import sys

from entrobus.utils import initialize_model, train_model

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import requests as req
from PIL import Image
from io import BytesIO
import json


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train and test Image Reconiton Task')

    parser.add_argument('--phase', help='Phase: Can be train or test, the default value is train',
                        default='train', type=str)

    # parameters required by train
    parser.add_argument('--train_dir', help='train images directory',
                        default=None, type=str)
    parser.add_argument('--val_dir', help='validate images directory',
                        default=None, type=str)
    parser.add_argument('--model_name', help='Model used to train. Such as resnet, vgg, inception',
                        default='squeezenet', type=str)
    parser.add_argument('--epochs', help='epochs used to train model',
                        default=10, type=int)
    parser.add_argument('--batch_size', help='batch_size used to train model',
                        default=4, type=int)
    parser.add_argument('--lr', help='learning rate used to train model',
                        default=0.001, type=float)
    parser.add_argument('--momentum', help='momentum used to train model',
                        default=0.9, type=float)
    parser.add_argument('--save_path', help='file path used to save model',
                        default='./model.pt', type=str)
    parser.add_argument('--gpu', help='file path used to save model',
                        default=False, type=bool)

    # parameters required by test
    parser.add_argument('--url', help='image url to be test',
                        default=None, type=str)
    parser.add_argument('--model_path', help='model to be used in test',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def train(args):
    # train parameters
    model_name = args.model_name
    train_dir = args.train_dir
    val_dir = args.val_dir
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    momentum = args.momentum
    save_path = args.save_path
    gpu = args.gpu

    if model_name == "inception":
        is_inception = True
    else:
        is_inception = False

    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    if train_dir is None:
        raise ValueError("train directory not define")

    train_data = ImageFolder(train_dir)
    num_classes = len(train_data.classes)

    model_ft, input_size = initialize_model(model_name, num_class=num_classes, extract_feature=False, pretrain=True)

    train_data = ImageFolder(train_dir, transform=transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=4)
    data_dataloader = {}
    data_dataloader['train'] = train_loader
    if val_dir:
        test_data = ImageFolder(val_dir, transform=transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))
        test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, num_workers=4)
        data_dataloader['val'] = test_loader

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_ft.parameters(), lr, momentum)
    model_ft, hist_acc, top1_accuracy, top5_accuracy = train_model(model_ft, data_dataloader, criterion, optimizer,
                                                                   epochs, device, is_inception=is_inception)
    classes = train_data.classes
    rlt = {}
    rlt['classes'] = classes

    rlt['parameters_dict'] = model_ft.cpu().state_dict()
    rlt['model_name'] = model_name
    rlt['top1'] = top1_accuracy
    rlt['top5'] = top5_accuracy
    torch.save(rlt, open(save_path, 'wb'))
    print("Success saved!")
    print("top1: {}".format(top1_accuracy))
    print("top5: {}".format(top5_accuracy))
    print()


def test(args):
    # test parameters
    url = args.url
    model_path = args.model_path

    if url is None or model_path is None:
        raise ValueError("url and model_path must be defined!")

    parameter_dict = torch.load(model_path)
    model_name = parameter_dict['model_name']
    num_classes = len(parameter_dict['classes'])
    classes = parameter_dict['classes']

    model_ft, input_size = initialize_model(model_name, num_classes, extract_feature=False, pretrain=True)
    model_ft.load_state_dict(parameter_dict['parameters_dict'])

    img_trans = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36"}
    response = req.get(url, headers=headers, stream=True)
    image = Image.open(BytesIO(response.content))
    image = image.convert("RGB")
    inputs = img_trans(image)
    inputs = inputs.unsqueeze(0)

    with torch.no_grad():
        model_ft.eval()
        outputs = model_ft(inputs)
        probs = torch.softmax(outputs, dim=1)
        probs = probs.view(-1)
        top5 = np.argsort(probs.numpy())[-5:].tolist()
        top5.reverse()
    rlt = json.dumps(list(zip(np.array(classes)[top5].tolist(), probs.numpy()[top5].tolist())))
    print(json.loads(rlt))
    return rlt


if __name__ == "__main__":
    args = parse_args()
    print('Called with args:')
    print(args)

    phase = args.phase

    if phase == 'train':
        train(args)
    elif phase == 'test':
        test(args)
    else:
        raise ValueError("Phase not defined. Please use 'train' or 'test'!")
