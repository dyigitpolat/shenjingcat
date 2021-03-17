from __future__ import print_function
import os

from torch._C import dtype

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from catSNN import spikeLayer, transfer_model, load_model, max_weight, normalize_weight, SpikeDataset ,fuse_bn_recursively, fuse_module
from utils import to_tensor

import torch.utils.data.dataloader
import torch.utils.data.dataset

from models.vgg import VGG, VGG_,CatVGG,CatVGG_
from models.vgg_onlyt import VGG_o, CatVGG_o

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def Initialize_trainable_pooling(feature):
    datashape = feature.shape
    for i in range(datashape[0]):
        for j in range(datashape[1]):
            feature[i][j] = torch.Tensor(0.25*np.ones((2,2)))
    return nn.Parameter(feature, requires_grad=True)

def train_(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for module in model.modules():
        # print(module)
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()
        if isinstance(module, nn.Dropout):
            module.eval()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).float(), target.to(device).float()
        #onehot = torch.nn.functional.one_hot(target, 10)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for module in model.modules():
        # print(module)
        if isinstance(module, nn.BatchNorm1d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        onehot = torch.nn.functional.one_hot(target, 10)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    

def test_(model, device, test_loader, verbose = False):
    model.eval()
    print("model eval()")

    preds = []
    outputs = []
    test_loss = 0
    correct = 0
    cur = 0
    first = True
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #onehot = torch.nn.functional.one_hot(target, 10)
            output = model(data)

            test_loss += F.l1_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.argmax(dim=1, keepdim=True).view_as(pred)).sum().item()
            cur += len(output)
            if(verbose and (cur % 50*len(output) == 0)): 
                print("acc: " + str(100 * correct / cur))


    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct

def test(model, device, test_loader, verbose = False, save_preds_outputs = False, save_path = "./out.npy"):
    model.eval()
    print("model eval()")

    test_loss = 0
    correct = 0
    cur = 0
    preds = []
    outputs = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # onehot = torch.nn.functional.one_hot(target, 10)
            output = model(data)
                
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            cur += len(output)
            if(verbose and (cur % 50*len(output) == 0)): 
                print("acc: " + str(100 * correct / cur))
                print("processed:", cur)
            
            if(save_preds_outputs): 
                for i in range(len(output)):
                    preds.append(pred[i].cpu().numpy()[0]) 
                    outputs.append(output[i].cpu().numpy().tolist()) 


    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if(save_preds_outputs): 
        preds_np = np.asarray(preds)
        outputs_np = np.asarray(outputs)
        np.save(save_path + ".preds.npy", preds_np)
        np.save(save_path, outputs_np)

    return correct

##
def get_cifar10_trainset_loader():
    pass

def get_cifar10_testset():
    pass

def get_cifar10_trainset_loader_with_snn_labels():
    pass

def set_trainable_base(model :VGG_o, is_trainable :bool, device :torch.device):
    model.cpu()

    model.features[0].requires_grad_(is_trainable)
    model.features[4].requires_grad_(is_trainable)
    model.features[10].requires_grad_(is_trainable)
    model.features[14].requires_grad_(is_trainable)
    model.features[20].requires_grad_(is_trainable)
    model.features[24].requires_grad_(is_trainable)
    model.features[28].requires_grad_(is_trainable)
    model.features[34].requires_grad_(is_trainable)
    model.features[38].requires_grad_(is_trainable)
    model.features[42].requires_grad_(is_trainable)
    model.features[48].requires_grad_(is_trainable)
    model.features[52].requires_grad_(is_trainable)
    model.features[56].requires_grad_(is_trainable)

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(is_trainable)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(is_trainable)
            # module.eval()

    # classification layer should always be frozen.
    model.classifier.requires_grad_(False) 
            
    model.to(device)
    pass

def set_trainable_support(model :VGG_o, is_trainable :bool, device :torch.device):
    model.cpu()

    for f in model.features[62]:
        if hasattr(f, 'weight'):
            f.requires_grad_(is_trainable)

    for module in model.modules():
        # print(module)
        if isinstance(module, nn.BatchNorm1d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(is_trainable)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(is_trainable)
            # module.eval()

    # except first support, always trainable.
    model.features[62][2].requires_grad_(True)
    
    # classification layer should always be frozen.
    model.classifier.requires_grad_(False) 
    
    model.to(device)
    pass

def freeze_base(model :VGG_o, device :torch.device):
    set_trainable_base(model, False, device)

def unfreeze_base(model :VGG_o, device :torch.device):
    set_trainable_base(model, True, device)

def freeze_support(model :VGG_o, device :torch.device):
    set_trainable_support(model, False, device)

def unfreeze_support(model :VGG_o, device :torch.device):
    set_trainable_support(model, True, device)

def augment_model(model :VGG_o, device :torch.device):
    model.cpu()

    # augment model
    model.features[62] = nn.Sequential(
        model.features[62], nn.Flatten(), 
        nn.Linear(512,512,bias=False), nn.ReLU(),
        nn.Linear(512,512,bias=False), nn.ReLU(),
        nn.Linear(512,256,bias=False), nn.ReLU(),
        nn.Linear(256,256,bias=False), nn.ReLU(),
        nn.Linear(256,256,bias=False), nn.ReLU(),
        nn.Linear(256,256,bias=False), nn.ReLU(),
        nn.Linear(256,256,bias=False), nn.ReLU(),
        nn.Linear(256,256,bias=False), nn.ReLU(),
        nn.Linear(256,256,bias=False), nn.ReLU(),
        nn.Linear(256,512,bias=False), nn.ReLU(),
        nn.Linear(512,512,bias=False), 
        nn.Linear(512,512,bias=False))
    
    for m in model.features[62]:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    model.to(device)
    pass

def transfer_base_params(dst_model :VGG_o, src_model :VGG_o, device :torch.device):
    dst_model.cpu()
    src_model.cpu()

    with torch.no_grad():
        dst_model.features[0].weight.copy_(src_model.features[0].weight)
        dst_model.features[0].bias.copy_(src_model.features[0].bias)

        dst_model.features[1].weight.copy_(src_model.features[1].weight)
        dst_model.features[1].bias.copy_(src_model.features[1].bias)
        dst_model.features[1].running_mean.copy_(src_model.features[1].running_mean)
        dst_model.features[1].running_var.copy_(src_model.features[1].running_var)
        dst_model.features[1].num_batches_tracked.copy_(src_model.features[1].num_batches_tracked)

        dst_model.features[4].weight.copy_(src_model.features[4].weight)
        dst_model.features[4].bias.copy_(src_model.features[4].bias)

        dst_model.features[5].weight.copy_(src_model.features[5].weight)
        dst_model.features[5].bias.copy_(src_model.features[5].bias)
        dst_model.features[5].running_mean.copy_(src_model.features[5].running_mean)
        dst_model.features[5].running_var.copy_(src_model.features[5].running_var)
        dst_model.features[5].num_batches_tracked.copy_(src_model.features[5].num_batches_tracked)

        dst_model.features[10].weight.copy_(src_model.features[10].weight)
        dst_model.features[10].bias.copy_(src_model.features[10].bias)

        dst_model.features[11].weight.copy_(src_model.features[11].weight)
        dst_model.features[11].bias.copy_(src_model.features[11].bias)
        dst_model.features[11].running_mean.copy_(src_model.features[11].running_mean)
        dst_model.features[11].running_var.copy_(src_model.features[11].running_var)
        dst_model.features[11].num_batches_tracked.copy_(src_model.features[11].num_batches_tracked)

        dst_model.features[14].weight.copy_(src_model.features[14].weight)
        dst_model.features[14].bias.copy_(src_model.features[14].bias)

        dst_model.features[15].weight.copy_(src_model.features[15].weight)
        dst_model.features[15].bias.copy_(src_model.features[15].bias)
        dst_model.features[15].running_mean.copy_(src_model.features[15].running_mean)
        dst_model.features[15].running_var.copy_(src_model.features[15].running_var)
        dst_model.features[15].num_batches_tracked.copy_(src_model.features[15].num_batches_tracked)

        dst_model.features[20].weight.copy_(src_model.features[20].weight)
        dst_model.features[20].bias.copy_(src_model.features[20].bias)

        dst_model.features[21].weight.copy_(src_model.features[21].weight)
        dst_model.features[21].bias.copy_(src_model.features[21].bias)
        dst_model.features[21].running_mean.copy_(src_model.features[21].running_mean)
        dst_model.features[21].running_var.copy_(src_model.features[21].running_var)
        dst_model.features[21].num_batches_tracked.copy_(src_model.features[21].num_batches_tracked)

        dst_model.features[24].weight.copy_(src_model.features[24].weight)
        dst_model.features[24].bias.copy_(src_model.features[24].bias)

        dst_model.features[25].weight.copy_(src_model.features[25].weight)
        dst_model.features[25].bias.copy_(src_model.features[25].bias)
        dst_model.features[25].running_mean.copy_(src_model.features[25].running_mean)
        dst_model.features[25].running_var.copy_(src_model.features[25].running_var)
        dst_model.features[25].num_batches_tracked.copy_(src_model.features[25].num_batches_tracked)

        dst_model.features[30].weight.copy_(src_model.features[30].weight)
        dst_model.features[30].bias.copy_(src_model.features[30].bias)

        dst_model.features[31].weight.copy_(src_model.features[31].weight)
        dst_model.features[31].bias.copy_(src_model.features[31].bias)
        dst_model.features[31].running_mean.copy_(src_model.features[31].running_mean)
        dst_model.features[31].running_var.copy_(src_model.features[31].running_var)
        dst_model.features[31].num_batches_tracked.copy_(src_model.features[31].num_batches_tracked)

        dst_model.classifier.weight.copy_(src_model.classifier.weight)
        dst_model.classifier.bias.copy_(src_model.classifier.bias)
    
    src_model.to(device)
    dst_model.to(device)
    pass

def train_on_snn_output(model :VGG_o, snn_output_npy :np.ndarray, batch_size, lr, step_gamma, epochs, log_interval, device :torch.device):
    model.to(device)
    
    outputs = snn_output_npy

    # pure cifar train
    transform_train_plain = transforms.Compose([
        transforms.ToTensor()
        ])
    trainset_plain = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train_plain)

    # pure cifar test
    testset = datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform_train_plain)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False)

    print(trainset_plain.data.shape)
    print(outputs.shape)
    images = torch.transpose(torch.tensor(trainset_plain.data).float()  / 255., 1, 3)
    images = torch.transpose(images, 2, 3)
    print(images.shape)

    # create tensor dataset with cifar images and snn output
    trainset_dataset = torch.utils.data.dataset.TensorDataset(images, torch.tensor(outputs))
    train_loader_no_aug_no_shuffle = torch.utils.data.dataloader.DataLoader(trainset_dataset, batch_size=batch_size, shuffle=True) #was shuffle=False

    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=step_gamma)
    correct_ = 0

    for epoch in range(1, epochs + 1):
        train_(log_interval, model, device, train_loader_no_aug_no_shuffle, optimizer, epoch)
        correct = test_(model, device, train_loader_no_aug_no_shuffle, verbose=True)
        test(model, device, test_loader)
        dyn_threshold = 1.5 * (50000 - correct_) / (epochs)
        if correct>correct_+dyn_threshold:
            correct_ = correct
            scheduler.step()
            print("step!")

def train_on_cifar10(model, batch_size, aug_k, lr, step_gamma, epochs, log_interval, device :torch.device):
    model.to(device)

    # cifar train with data aug
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=6),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            AddGaussianNoise(std=0.01)
            ])

    trainset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    for i in range(aug_k):
        im_aug = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomCrop(32, padding = 6),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.01)
        ])
        trainset = trainset + datasets.CIFAR10(root='./data', train=True, download=True, transform=im_aug)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True)
    
    # pure cifar test
    transform_train_plain = transforms.Compose([
        transforms.ToTensor()
        ])
    testset = datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform_train_plain)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=step_gamma)
    correct_ = 0
    for epoch in range(1, epochs + 1):
        train(log_interval, model, device, train_loader, optimizer, epoch)
        #test(model, device, train_loader, verbose=True)
        correct = test(model, device, test_loader)
        
        dyn_threshold = 1.2 * (10000 - correct_) / (epochs)
        if correct>correct_+dyn_threshold:
            correct_ = correct
            scheduler.step()
            print("step!")
    pass

def validate_base_vs_tuned(tuned_model :VGG_o, base_model_path :str, device :torch.device):
    transform_train_plain = transforms.Compose([
        transforms.ToTensor()
        ])
    testset = datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform_train_plain)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False)
    try:
        print("original performance:")
        orig_model = create_base()
        orig_model.load_state_dict(torch.load(base_model_path), strict=False)
        orig_model.to(device)
        orig_model.eval()
        test(orig_model, device, test_loader)
    except:
        pass

    transfer_base_params(orig_model, tuned_model, device)

    print("validating tuned performance")
    orig_model.eval()
    test(orig_model, device, test_loader)
def create_base():
    return VGG_o('o', clamp_max=1,bias =True)

def create_base_from(model :VGG_o, device :torch.device):
    new_base_model = VGG_o('o', clamp_max=1,bias =True)
    transfer_base_params(new_base_model, model, device)
    return new_base_model

def save_as_new_base(model :VGG_o, new_base_model_path :str, device :torch.device):
    new_base_model = create_base_from(model, device)
    torch.save(new_base_model, new_base_model_path)
    pass

def evaluate_on_snn(base_model :VGG_o, T :int, generate_output_npy :bool, save_path :str, device :torch.device):
    # pure cifar train
    transform_train_plain = transforms.Compose([
        transforms.ToTensor()
        ])
    trainset_plain = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train_plain)

    # pure cifar test
    testset = datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform_train_plain)
        
    # SNN related dataset
    snn_dataset = SpikeDataset(testset, T = T)
    snn_loader = torch.utils.data.DataLoader(snn_dataset, batch_size=10, shuffle=False) #was shuffle=False
    snn_loader_shuff = torch.utils.data.DataLoader(snn_dataset, batch_size=10, shuffle=True) #was shuffle=False

    snn_dataset_trainset = SpikeDataset(trainset_plain, T = T)
    snn_loader_trainset = torch.utils.data.DataLoader(snn_dataset_trainset, batch_size=10, shuffle=False) #was shuffle=False

    # Initialize SNN model
    snn_model = CatVGG_o('o', T,bias =True).to(device)

    print("model ready")
    base_model = fuse_bn_recursively(base_model)
    print("model fused")
    transfer_model(base_model, snn_model)
    print("model transferred")

    with torch.no_grad():
        normalize_weight(snn_model.features, quantize_bit=32)
        print("model weight normalized")
    
    if(generate_output_npy):
        test(snn_model, device, snn_loader_trainset, save_preds_outputs=True, verbose = True, save_path=save_path)
    else:
        test(snn_model, device, snn_loader_shuff, verbose = True) #old
    print("snn model test ok")
    pass
##

def retrieve_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 LeNet Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--resume', type=str, default=None, metavar='RESUME',
                        help='Resume model from checkpoint')
    parser.add_argument('--T', type=int, default=100, metavar='N',
                        help='SNN time window')
    parser.add_argument('--k', type=int, default=0, metavar='N',
                        help='Data augmentation')
    parser.add_argument('--loops', type=int, default=0, metavar='N',
                        help='loops count')
    parser.add_argument('--from-scratch', type=int, default=0, metavar='N',
                        help='to train from scratch or load model')
    
    args = parser.parse_args()
    
    return args