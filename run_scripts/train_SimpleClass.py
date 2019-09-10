import os, argparse, sys
sys.path.append('..')
from pathlib import Path
import torch
from torch import nn, optim
from SimpleClass.model import Net
from SimpleClass.DataLoader import loadCIFAR10
from SimpleClass.evaluation import eval_test_accuracy

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='run_001.pth')
opt = parser.parse_args()
model_path = Path('/notebooks/DockerShared/MINE/models/SimpleClass')
model_name = Path(opt.model_name)
save_pathname = model_path / model_name

trainloader, testloader, classes = loadCIFAR10()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

logging = False
for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if logging and i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
torch.save(net.state_dict(), str(save_pathname))
eval_test_accuracy(net, testloader, device, logging=True)