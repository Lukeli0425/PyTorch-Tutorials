# 2022-2-8 luke
# demo of classification on mnist
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from matplotlib import pyplot as plt
import numpy as np

# utils fuctions
def one_hot(label,depth):
    onehot = torch.zeros(label.size(0),depth)
    idx = torch.LongTensor(label).view(-1,1)
    onehot.scatter_(dim=1,index=idx,value=1)
    return onehot

def visualize_minst(image,label,name):
    plt.figure(name.title())
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(image[i][0]*0.3801+0.1307,cmap='gray',interpolation='none')
        plt.title(name+f": {label[i]}")
    plt.show()

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # linear layers
        self.fc1 = nn.Linear(28*28,256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,10)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        return y


if __name__ == "__main__":
    # load mnist dataset
    train_dataset = torchvision.datasets.MNIST(root='./mnist_data/train_dataset',train=True,download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        (0.1307,),(0.3801)
                                                    )
                                                ]))
    test_dataset = torchvision.datasets.MNIST(root='./mnist_data/test_dataset/',train=False,download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        (0.1307,),(0.3801)
                                                    )
                                                ]))
    batch_size = 200
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

    x,y = next(iter(train_loader))
    print(x.shape)
    print(y.shape)
    visualize_minst(x,y,'label')

    # build model
    model = Network()

    # train model
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
    history = []
    for epoch in range(0,3):
        for batch_idx,(x,y) in enumerate(train_loader):
            # reshape: [b,1,28,28] -> [b,28*28]
            x = x.view(x.size(0), 28*28)
            output = model(x)
            y_onehot = one_hot(y,depth=10)
            loss = F.cross_entropy(output, y_onehot)
            # loss = F.mse_loss(output, y_onehot)
            optimizer.zero_grad()
            loss.backward() # compute gradient
            optimizer.step() # update model parameters
            history.append(loss.item())
            if batch_idx % 10 == 0:
                N = len(train_loader)
                print(f"[{epoch}:{batch_idx}/{N}] {loss}")

# visualize training process
plt.figure('Loss')
plt.plot(range(len(history)),history)
plt.xlabel('step')
plt.ylabel('loss')
plt.show()

# test model on test dataset
total_correct = 0
total_num = len(test_loader.dataset)
for x,y in test_loader:
    x = x.view(x.size(0),28*28)
    output = model(x)
    pred = output.argmax(dim=1)
    corect = pred.eq(y).sum().float().item() # numer of correct predictions in current batch
    total_correct += corect
acc = total_correct / total_num
print(f"test accuracy: {acc}")
x,y = next(iter(test_loader))
output = model(x.view(x.size(0),28*28))
pred = output.argmax(dim=1)
visualize_minst(x,pred,'prediction')