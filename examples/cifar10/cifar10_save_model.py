from __future__ import print_function
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
from catSNN import spikeLayer, transfer_model, load_model, max_weight, normalize_weight, SpikeDataset ,fuse_bn_recursively, fuse_module
from utils import to_tensor

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

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        onehot = torch.nn.functional.one_hot(target, 10)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, verbose = False, batch_size = 10):
    model.eval()
    print("model eval()")

    preds = []
    outputs = []
    test_loss = 0
    correct = 0
    cur = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            onehot = torch.nn.functional.one_hot(target, 10)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            cur += batch_size
            if(verbose): 
                for i in range(batch_size):
                    preds.append(pred[i].cpu().numpy()[0]) 
                    outputs.append(output[i].cpu().numpy().tolist()) 
                #print(preds)
                #print(outputs)
                print("acc: " + str(100 * correct / cur))


    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if(verbose): 
        preds_np = np.asarray(preds)
        outputs_np = np.asarray(outputs)
        np.save("examples/cifar10/preds.npy", preds_np)
        np.save("examples/cifar10/outputs.npy", outputs_np)

    return correct



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 LeNet Example')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
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
    parser.add_argument('--k', type=int, default=10, metavar='N',
                        help='Data augmentation')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=6),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.01)
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
        ])

    trainset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    train_loader_ = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    # pure cifar
    transform_train_plain = transforms.Compose([
        transforms.ToTensor()
        ])

    trainset_plain = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train_plain)
    snn_dataset_no_aug_no_shuffle = SpikeDataset(trainset_plain, T = args.T)
    snn_loader_no_aug_no_shuffle = torch.utils.data.DataLoader(snn_dataset_no_aug_no_shuffle, batch_size=10, shuffle=False) #was shuffle=False
       
    for i in range(args.k):

        im_aug = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomCrop(32, padding = 6),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.01)
        ])
        trainset = trainset + datasets.CIFAR10(root='./data', train=True, download=True, transform=im_aug)
    
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True)


    # train_loader = torch.utils.data.DataLoader(
    #     trainset, batch_size=256+512, shuffle=True)   

    testset = datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False)

    snn_dataset = SpikeDataset(testset, T = args.T)
    snn_loader = torch.utils.data.DataLoader(snn_dataset, batch_size=10, shuffle=False) #was shuffle=False

    from models.vgg import VGG, VGG_, CatVGG, CatVGG_
    
    # model = VGG('VGG16', clamp_max=1.0, quantize_bit=32,bias =False).to(device)
    # snn_model = CatVGG('VGG16', args.T,bias =True).to(device)

    #Trainable pooling
    model = VGG_('VGG16_', clamp_max=1.0, quantize_bit=32,bias =True).to(device)
    snn_model = CatVGG_('VGG16_', args.T,bias =True).to(device)

    if args.resume != None:
        model.load_state_dict(torch.load(args.resume), strict=False)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    
    loaded_model = torch.load("examples/cifar10/model_clamped.bin").to(device)
    # loaded_model.cpu()

    # #####
    # # copy loaded model
    # print(model.features[0].weight)
    # print(loaded_model.features[0].weight)
    # model.features[0].weight.data.copy_(loaded_model.features[0].weight.data)
    # model.features[2].weight.data.copy_(loaded_model.features[4].weight.data)
    # model.features[6].weight.data.copy_(loaded_model.features[10].weight.data)
    # model.features[8].weight.data.copy_(loaded_model.features[14].weight.data)
    # model.features[12].weight.data.copy_(loaded_model.features[20].weight.data)
    # model.features[14].weight.data.copy_(loaded_model.features[24].weight.data)
    # model.features[16].weight.data.copy_(loaded_model.features[28].weight.data)
    # model.features[20].weight.data.copy_(loaded_model.features[34].weight.data)
    # model.features[22].weight.data.copy_(loaded_model.features[38].weight.data)
    # model.features[24].weight.data.copy_(loaded_model.features[42].weight.data)
    # model.features[28].weight.data.copy_(loaded_model.features[48].weight.data)
    # model.features[30].weight.data.copy_(loaded_model.features[52].weight.data)
    # model.features[32].weight.data.copy_(loaded_model.features[56].weight.data)
    # model.classifier.weight.data.copy_(loaded_model.classifier.weight.data)
    # print(model.features[0].weight)

    # model = model.to(device)
    # loaded_model.to(device)
    # #####
    # #####

    # model.eval()
    # test(model, device, test_loader, verbose=True, batch_size=100)
    test(loaded_model, device, torch.utils.data.DataLoader(trainset_plain, batch_size=1024, shuffle=False), verbose=True, batch_size=1024)

    exit();

    try:
        correct_ = 0
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, train_loader_)
            correct = test(model, device, test_loader)
            if correct>correct_:
                correct_ = correct
            scheduler.step()
        
        torch.save(model, "examples/cifar10/model_clamped.bin")

    except:
        correct_ = 0
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, train_loader_)
            correct = test(model, device, test_loader)
            if correct>correct_:
                correct_ = correct
            scheduler.step()
        
        torch.save(model, "examples/cifar10/model_clamped.bin")

    print("model ready")
    model = fuse_bn_recursively(model)
    print("model fused")
    transfer_model(model, snn_model)
    print("model transferred")

    with torch.no_grad():
        normalize_weight(snn_model.features, quantize_bit=32)
        print("model weight normalized")

    #test(snn_model, device, snn_loader, verbose = True) #old
    test(snn_model, device, snn_loader_no_aug_no_shuffle, verbose = True)
    print("snn model test ok")



if __name__ == '__main__':
    main()
