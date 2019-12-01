# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Import relevant packages
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import csv
import os
import time
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)


# %%
# Hyperparameters and config
parser = argparse.ArgumentParser()
parser.add_argument("--dim")
parser.add_argument("--lr")
parser.add_argument("--epochs")
parser.add_argument("--debug", action='store_true')
parser.add_argument("--disablecuda", action='store_true')
args = parser.parse_args()

INPUT_DIM = int(args.dim) if args.dim else 128
LR = float(args.lr) if args.lr else 0.0001
NUM_EPOCHS = int(args.epochs) if args.epochs else 12
DEBUG = args.debug
# MAX_NUM_IMAGES_PER_DATASET = 3664  # 2 x size of smaller dataset
train_test_ratio = 0.8

DISABLE_CUDA = args.disablecuda
MODEL_NAME = "CNN v2.8.1 Don't call model.eval()"


# %%
# Select accelerator device
def get_default_device():
    """Returns device, is_cuda (bool)."""
    if not DISABLE_CUDA and torch.cuda.is_available():
        print("Running on CUDA!")
        return torch.device('cuda'), True
    else:
        print("Running on CPU ;(")
        return torch.device('cpu'), False


device, using_cuda = get_default_device()


# %%
# Declare important file paths
project_path = os.path.abspath('')
data_path = project_path + '/data/ldrd-and-raise-datasets/image-folder'
model_path = project_path + '/models/' + MODEL_NAME + \
    ' ~ dim={}, lr={}, epochs={}, cuda={}.csv'.format(
        INPUT_DIM, LR, NUM_EPOCHS, using_cuda) + '-model.pth'


# %%
def obtain_data(input_dim):
    # Transform the data
    transform = transforms.Compose([
        transforms.Resize((input_dim, input_dim)),  # resize to square
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create training/testing dataloaders
    full_set = datasets.ImageFolder(root=data_path, transform=transform)
    train_size = int(train_test_ratio * len(full_set))
    val_size = int((len(full_set) - train_size) / 2)
    test_size = len(full_set) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(
        full_set, [train_size, val_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False)

    return train_loader, val_loader


# %%
# # Load data into memory to elimate read bottleneck
# TODO this is called before every epoch so it's not useful. Refactor train_model to fix.
def load_data_into_memory(data_loader):
    output = []
    for data in data_loader:
        inputs = data[0].to(device, non_blocking=True)
        labels = data[1].to(device, non_blocking=True)
        output.append((inputs, labels))
    return output


def one_shot_data_generator(data_loader):
    for i, data in enumerate(data_loader):
        inputs = data[0].to(device, non_blocking=True)
        labels = data[1].to(device, non_blocking=True)
        yield (inputs, labels)


# %%
# Declare our model architecture
def declare_model(input_dim):
    class ConvNet(nn.Module):  # Convolutional Neural Network
        def __init__(self):
            super(ConvNet, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=4),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                # nn.Dropout(0.1),
                nn.MaxPool2d(kernel_size=5, stride=2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=4),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # nn.Dropout(0.1),
                nn.MaxPool2d(kernel_size=5, stride=2))
            self.layer3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=6),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                # nn.Dropout(0.1),
                nn.MaxPool2d(kernel_size=5, stride=2))
            self.layer4 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=6),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                # nn.Dropout(0.1),
                nn.MaxPool2d(kernel_size=int(input_dim/16), stride=2))
            # TODO this doesn't like intput_dim that aren't divisible by 8 (e.g. 650)
            self.fc1 = nn.Sequential(
                nn.Linear(256, 64),
                # nn.Dropout(0.5),
                nn.ReLU())
            self.fc2 = nn.Sequential(
                nn.Linear(64, 1),
                # nn.Dropout(0.5),
                nn.Sigmoid())

        def forward(self, x):
            if DEBUG:
                print(x.shape)
            out = self.layer1(x)
            if DEBUG:
                print(out.shape)
            out = self.layer2(out)
            if DEBUG:
                print(out.shape)
            out = self.layer3(out)
            if DEBUG:
                print(out.shape)
            out = self.layer4(out)
            if DEBUG:
                print(out.shape)
            out = out.reshape(out.size(0), -1)
            if DEBUG:
                print(out.shape)
            out = self.fc1(out)
            if DEBUG:
                print(out.shape)
            out = self.fc2(out)
            if DEBUG:
                print(out.shape)
            return out

    model = ConvNet()
    model.to(device)
    return model


# %%
def train_model(model, loss_fn, optimizer, train_loader, val_loader, num_epochs):
    loss_list = []
    time_list = []
    train_accuracy_list = []
    val_accuracy_list = []
    t = torch.Tensor([0.5]).to(device)  # 0.5 acts as threshold
    highest_acc = 0.0

    torch.backends.cudnn.benchmark = True  # make training faster on Cuda

    start_time = time.time()

    model.train()  # switch to train mode
    for epoch in range(num_epochs):
        # Train the model

        running_loss = 0.0
        train_correct = train_total = 0
        for i, (inputs, labels) in enumerate(one_shot_data_generator(train_loader)):
            # if i > MAX_NUM_IMAGES_PER_DATASET:
            #     break

            labels = labels.view(-1, 1)

            probs = model(inputs)

            outputs = (probs > t).float() * 1  # obtain train accuracies
            train_total += len(outputs)
            # normalize batch size
            train_correct += (outputs == labels.float()
                              ).float().sum() / len(outputs)

            loss = loss_fn(probs, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # if (i + 1) % 10 == 0:
            #     print ('# Images: {:} | Time (m): {:.3f} | Loss: {:.6f} '.format(i + 1, (time.time() - start_time)/60, running_loss / (i + 1)))
        train_accuracy = train_correct / train_total

        # Test current version of model to obtain accuracy
        # model.eval()  # switch to evaluation mode
        val_correct = val_total = 0
        with torch.no_grad():
            for (inputs, labels) in one_shot_data_generator(val_loader):
                labels = labels.view(-1, 1)

                probs = model(inputs)
                outputs = (probs > t).float() * 1
                val_total += len(outputs)
                # normalize batch size
                val_correct += (outputs == labels.float()
                                ).float().sum() / len(outputs)
        val_accuracy = val_correct / val_total

        if val_accuracy > highest_acc:  # save highest accuracy model
            highest_acc = val_accuracy
            torch.save(model.state_dict(), model_path)

        elapsed_time = (time.time() - start_time)/60
        time_list.append(elapsed_time)
        loss_list.append(running_loss)
        train_accuracy_list.append(train_accuracy)
        val_accuracy_list.append(val_accuracy)
        print('Epoch: {:} | Time (m): {:.6f} | Loss: {:.6f} | Train Accuracy: {:.8%} | Validation Accuracy: {:.8%}'.format(
            epoch, elapsed_time, running_loss, train_accuracy, val_accuracy))

    return time_list, loss_list, train_accuracy_list, val_accuracy_list


# %%
def write_experiment_results_to_file(filename, results_dict):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(results_dict.keys())
        num_rows = len(list(results_dict.values())[0])
        for i in range(num_rows):
            row = []
            for key in results_dict.keys():
                row.append(float(results_dict[key][i]))
            writer.writerow(row)
        print('Wrote {} rows to file.'.format(num_rows))


# %%
def run_experiment(input_dim, lr, num_epochs):
    train_loader, val_loader = obtain_data(input_dim)

    model = declare_model(input_dim)

    loss_fn = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    results_filename = 'experiments/' + MODEL_NAME + \
        ' ~ dim={}, lr={}, epochs={}, cuda={}.csv'.format(
            input_dim, lr, num_epochs, using_cuda)
    print(results_filename)

    time_list, loss_list, train_accuracy_list, val_accuracy_list = train_model(
        model, loss_fn, optimizer, train_loader, val_loader, num_epochs=num_epochs)
    results_dict = {"time (m)": time_list, "loss": loss_list,
                    "train accuracy": train_accuracy_list, "val accuracy": val_accuracy_list}
    write_experiment_results_to_file(results_filename, results_dict)

    return time_list, loss_list, train_accuracy_list, val_accuracy_list


# %%
# run the experiment
time_list, loss_list, train_accuracy_list, val_accuracy_list = run_experiment(
    INPUT_DIM, LR, NUM_EPOCHS)
