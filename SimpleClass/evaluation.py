import torch

def eval_test_accuracy(modelnet, testloader, device, logging=True):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = modelnet(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    if logging:
        #print('Accuracy of the network on the 10000 test images: {:.2%}'.format(accuracy))
        print(accuracy)
        