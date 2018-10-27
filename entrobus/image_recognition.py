import argparse
import numpy as np
import sys

from .utils import initialize_model, train_model

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


class JsonParser(object):
    def __init__(self, config):
        self.parameter_dict = {
            'phase': 'train',

            # require parameter
            'train_dir': None,
            'url': None,
            'model_path': None,

            # option parameters
            'val_dir': None,
            'model_name': "squeezenet",
            'epochs': 10,
            'batch_size': 4,
            'lr': 0.001,
            'momentum': 0.9,
            'gpu': False,
            'save_path': './model_default.pt'
        }
        config_dict = json.loads(config)
        for k, v in config_dict.items():
            self.parameter_dict[k] = v


def train(config):
    args = JsonParser(config)
    print("The parameters set for training:")
    print('{')
    for k, v in args.parameter_dict.items():
        print("\t{:<15} : {}".format(k, v))
    print('}')
    # train parameters
    model_name = args.parameter_dict['model_name']
    train_dir = args.parameter_dict['train_dir']
    val_dir = args.parameter_dict['val_dir']
    epochs = args.parameter_dict['epochs']
    batch_size = args.parameter_dict['batch_size']
    lr = args.parameter_dict['lr']
    momentum = args.parameter_dict['momentum']
    save_path = args.parameter_dict['save_path']
    gpu = args.parameter_dict['gpu']



    if args.parameter_dict['phase'] != 'train':
        raise ValueError("phase should be train.")

    if train_dir is None:
        raise ValueError("train directory not define")

    if model_name == "inception":
        is_inception = True
    else:
        is_inception = False

    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

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

    rlt = {
        'model_path': save_path,
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy
    }
    return json.dumps(rlt)


def test(config):
    args = JsonParser(config)
    # test parameters
    url = args.parameter_dict['url']
    model_path = args.parameter_dict['model_path']
    if args.parameter_dict['phase'] != 'test':
        raise ValueError("phase should be test.")
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
    # print(json.loads(rlt))
    return rlt