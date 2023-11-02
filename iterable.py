# !pip install wandb
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch.utils.data
import numpy as np
import wandb
import torchvision.models as models

wandb.login()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)


torch.manual_seed(181988)
if torch.cuda.is_available():
    torch.cuda.manual_seed(181988)


batch_size =32
learning_rate = 0.02
momentum =.9
epochs = 10
name_net_exp = ['MLP','fc_hl_bigvalues','fc_hl_1layers','fc_hl_values_and_2layers','fc_hl_bigvalues_and_2layers',
                'fc_hl_less_values','fc_hl_1layer_less','fc_hl_1layerless_smallvalues','cnn_lessvalues','cnn_lessvalues_LR_MP',
                'CNN_LeNet-5','cnn_lessvalues_resnet']
project_name = "TP3"


for name in name_net_exp:
    experiment_name = name

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    targets_ = trainset.targets
    train_idx, val_idx = train_test_split(np.arange(len(targets_)), test_size=0.2, stratify=targets_)
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

    trainloader = torch.utils.data.DataLoader(trainset, sampler=train_sampler,batch_size=batch_size, num_workers=2)
    valloader = torch.utils.data.DataLoader(trainset, sampler=val_sampler,batch_size=batch_size, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    if experiment_name == "MLP":
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32*32*3, 32*32*3)
                self.fc21 = nn.Linear(32*32*3, 32*32)
                self.fc22 = nn.Linear(32*32, 120)
                self.fc23 = nn.Linear(120, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10) # termina con 10 para quedarse con la mejor

            def forward(self, x):
                x = torch.flatten(x, 1) # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc21(x))
                x = F.relu(self.fc22(x))
                x = F.relu(self.fc23(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

    elif experiment_name == "fc_hl_bigvalues":
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32*32*3, 32*32*3)
                self.fc21 = nn.Linear(32*32*3, 32*32)
                self.fc22 = nn.Linear(32*32, 150)
                self.fc23 = nn.Linear(150, 150)
                self.fc2 = nn.Linear(150, 100)
                self.fc3 = nn.Linear(100, 10) # termina con 10 para quedarse con la mejor

            def forward(self, x):
                x = torch.flatten(x, 1) # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc21(x))
                x = F.relu(self.fc22(x))
                x = F.relu(self.fc23(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

    elif experiment_name == "fc_hl_1layers":
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32*32*3, 32*32*3)
                self.fc21 = nn.Linear(32*32*3, 32*32)
                self.fc22 = nn.Linear(32*32, 300)
                self.fc23 = nn.Linear(300, 200)
                self.fc_hidden = nn.Linear(200, 120)  # Nueva capa oculta con 120 nodos
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x):
                x = torch.flatten(x, 1) # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc21(x))
                x = F.relu(self.fc22(x))
                x = F.relu(self.fc23(x))
                x = F.relu(self.fc_hidden(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

    elif experiment_name == "fc_hl_bigvalues_and_2layers":
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32*32*3, 32*32*3)
                self.fc21 = nn.Linear(32*32*3, 32*32)
                self.fc22 = nn.Linear(32*32, 300)
                self.fc23 = nn.Linear(300, 200)
                self.fc_hidden12 = nn.Linear(200, 200)  # Nueva capa oculta con 120 nodos
                self.fc2 = nn.Linear(200, 120)
                self.fc_hidden23 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x):
                x = torch.flatten(x, 1) # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc21(x))
                x = F.relu(self.fc22(x))
                x = F.relu(self.fc23(x))
                x = F.relu(self.fc_hidden12(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc_hidden23(x))
                x = self.fc3(x)
                return x

    elif experiment_name == "fc_hl_less_values":
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32*32*3, 32*32*3)
                self.fc21 = nn.Linear(32*32*3, 32*32)
                self.fc22 = nn.Linear(32*32, 50)
                self.fc23 = nn.Linear(50, 50)
                self.fc2 = nn.Linear(50, 25)
                self.fc3 = nn.Linear(25, 10) # termina con 10 para quedarse con la mejor

            def forward(self, x):
                x = torch.flatten(x, 1) # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc21(x))
                x = F.relu(self.fc22(x))
                x = F.relu(self.fc23(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

    elif experiment_name == "fc_hl_1layer_less":
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32*32*3, 32*32*3)
                self.fc21 = nn.Linear(32*32*3, 32*32)
                self.fc22 = nn.Linear(32*32, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10) # termina con 10 para quedarse con la mejor

            def forward(self, x):
                x = torch.flatten(x, 1) # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc21(x))
                x = F.relu(self.fc22(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

    elif experiment_name == "fc_hl_1layerless_smallvalues":
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32*32*3, 32*32*3)
                self.fc21 = nn.Linear(32*32*3, 32*32)
                self.fc22 = nn.Linear(32*32, 50)
                self.fc2 = nn.Linear(50, 25)
                self.fc3 = nn.Linear(25, 10) # termina con 10 para quedarse con la mejor

            def forward(self, x):
                x = torch.flatten(x, 1) # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc21(x))
                x = F.relu(self.fc22(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
            
    # Convolusionales

    elif experiment_name == "cnn_lessvalues":
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6,kernel_size = 5, stride = 1, padding = 1)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1, padding = 1)
                self.fc1 = nn.Linear(16 * 5 * 5, 50)
                self.fc21 = nn.Linear(50, 50)
                self.fc22 = nn.Linear(50, 50)
                self.fc23 = nn.Linear(50, 50)
                self.fc2 = nn.Linear(50, 25)
                self.fc3 = nn.Linear(25, 10)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = torch.flatten(x, 1) # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc21(x))
                x = F.relu(self.fc22(x))
                x = F.relu(self.fc23(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

    elif experiment_name == "CNN_LeNet-5":
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
                self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
                self.conv2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5)
                self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
                self.fc1 = nn.Linear(1728, 256)  # Adjusted output size to 256
                self.fc2 = nn.Linear(256, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x):
                x = self.pool1(F.relu(self.conv1(x)))
                x = self.pool2(F.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)  # Flatten the tensor
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

    elif experiment_name == "cnn_lessvalues_resnet":
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.resnet18 = models.resnet18(pretrained=True)
                num_ftrs = self.resnet18.fc.in_features
                self.resnet18.fc = nn.Linear(num_ftrs, 10)

            def forward(self, x):
                return self.resnet18(x)
            

    net = Net()
    net.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)


    # implementar json dump para que se arme solo el archivo
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,
        name = experiment_name,
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "momentum": momentum,
            "batch_size": batch_size,
            "epochs": epochs,
        }
    )


    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        train_correct =0
        total = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            #inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')


            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            train_correct += (predicted == labels).sum().item()


        # End of test section
        # Val section
        train_accuracy = 100 * train_correct / total
        running_loss = running_loss/total

        val_correct = 0
        total =0
        val_loss =0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in valloader:
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += criterion(outputs, labels).item()

        # End of test section

        val_accuracy = 100 * val_correct /total
        val_loss = val_loss / total

        wandb.log({ "train_accuracy": train_accuracy, "val_accuracy": val_accuracy, "train_loss": running_loss,
                "val_loss": val_loss})

    print('Finished Training')


    wandb.finish()