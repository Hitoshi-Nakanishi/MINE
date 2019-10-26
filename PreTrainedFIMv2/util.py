from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

class CIFAR10Worker:
    checkpoint_path = Path('/notebooks/DockerShared/MINE/models/PreTrainedv2')

    def __init__(self, device, model, criterion, params):
        """
        params ... log_interval, 
        """
        self.device = device
        self.model = model
        self.criterion = criterion
        self.params = params

    def load_data_loader(self):
        rootdir = '../../data'
        transform = transforms.Compose(
            [transforms.ToTensor(), 
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = CIFAR10(root=rootdir, train=True, download=True, transform=transform)
        self.trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)
        testset = CIFAR10(root=rootdir, train=False, download=True, transform=transform)
        self.testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return self

    def set_save_path(self, filename):
        self.save_path = CIFAR10Worker.checkpoint_path / Path(filename)
        return self
    
    def load_chckpt(self):
        self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))
        return self

    def save_chckpt(self):
        torch.save(self.model.state_dict(), self.save_path)
        return self

    def train(self, epoch, optimizer, default_env = True):
        if default_env:
            self.model.not_eval_FIM()
            self.model.train()
        train_loss, correct, total = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            targets_hat, _ = self.model(inputs)
            loss = self.criterion(targets_hat, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()            
            _, predicted = targets_hat.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % self.params['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(self.trainloader.dataset),
                    100. * batch_idx / len(self.trainloader),
                    loss.item() / len(inputs)))
        acc = correct/total
        print('Accuracy: {:.3%}'.format(acc))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(self.trainloader.dataset)))

    def test(self, default_env = True):
        if default_env:
            self.model.not_eval_FIM()
            self.model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets_hat, _ = self.model(inputs)
                loss = self.criterion(targets_hat, targets)
                test_loss += loss.item()
                _, predicted = targets_hat.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
                
        test_loss /= len(self.testloader.dataset)
        acc = correct/total
        print('Test-set loss: {:.4f}'.format(test_loss))
        print('Accuracy: {:.3%}'.format(acc))

    def evaluate_FIM(self, epoch, optimizer, layerIds = [], default_env = True):
        if default_env:
            self.model.eval()
            self.model.eval_FIM()
        for layerId in layerIds:
            self.model.inactivate_parameters_ex_specific_layer(layerId)
        
        def loss_for_FIM(y_hat, logvars, y):
            kl_div = torch.sum((10.*torch.exp(logvars) - logvars - 1))
            beta = 1
            cross_entropy = F.cross_entropy(y_hat, y)
            # w_squared = self.model.w_squared
            #return cross_entropy
            #return kl_div
            return  cross_entropy + beta*kl_div
            #return  cross_entropy + beta*kl_div# + w_squared
            #return  kl_div + w_squared

        train_loss, correct, total = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            targets_hat, logvars = self.model(inputs)
            logvars = torch.squeeze(torch.cat(logvars,dim=1))
            loss = loss_for_FIM(targets_hat, logvars, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()            
            total += targets.size(0)
            if batch_idx % 625 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(self.trainloader.dataset),
                    100. * batch_idx / len(self.trainloader),
                    loss.item() / len(inputs)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(self.trainloader.dataset)))