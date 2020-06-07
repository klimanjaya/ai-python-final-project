# Greetings
name = input("Enter your name: ")
print("Hello there, {}!".format(name.title()))


# Imports
import numpy as np
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from workspace_utils import active_session


# Loading data
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transforms for the training, validation, and testing sets
data_transforms = {'train': transforms.Compose([transforms.RandomRotation(45),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
                    'valid': transforms.Compose([transforms.Resize(275),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
                    'test': transforms.Compose([transforms.Resize(300),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])}


# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform = data_transforms['train'])
valid_data = datasets.ImageFolder(valid_dir, transform = data_transforms['valid'])
test_data = datasets.ImageFolder(test_dir, transform = data_transforms['test'])

image_datasets = [train_data, valid_data, test_data]

# Using the image datasets and the trainforms, define the dataloaders
trainloaders = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
validloaders = torch.utils.data.DataLoader(valid_data, batch_size = 64)
testloaders = torch.utils.data.DataLoader(test_data, batch_size = 64)


# Ask for model architecture and load pretrained model
chosen_arc = input("Please enter one of these model architectures, vgg19 or densenet121:  ")

if chosen_arc == 'vgg19':
    model = models.vgg19(pretrained=True)
    print("You chose pretrained VGG-19.")
elif chosen_arc == 'densenet121':
    model = models.densenet121(pretrained=True)
    print("You chose pretrained DenseNet-121.")
else:
    chosen_arc = input("Your entry is invalid. Please type in vgg19 or densenet121:  ")
    if chosen_arc == 'vgg19':
        model = models.vgg19(pretrained=True)
        print("VGG-19 is successfully chosen.")
    elif chosen_arc == 'densenet121':
        model = models.densenet121(pretrained=True)
        print("DenseNet-121 is successfully chosen.")
    else:
        print("You have not chosen the correct model. You will exit this program now. Goodbye.")
        exit()


# Ask for GPU or CPU
gpu = input("\nWould you like to train the model on GPU? (Y/N)  ")
if (gpu in ('Y', 'y')) or (gpu in ('Yes', 'yes')):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is enabled for model training.")
    else:
        device = torch.device("cpu")
        print("GPU is not available. Your model will be trained on CPU.")

elif (gpu in ('N', 'n')) or (gpu in ('No', 'no')):
    device = torch.device("cpu")
    print("Model training will run on CPU.")

else:
    gpu1 = input("Your answer is invalid. Would you like to train the model on GPU? (Y/N)  ")
    if (gpu1 in ('Y', 'y')) or (gpu1 in ('Yes', 'yes')):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU is enabled for model training.")
        else:
            device = torch.device("cpu")
            print("GPU is not available. Your model will be trained on CPU.")

    elif (gpu1 in ('N', 'n')) or (gpu1 in ('No', 'no')):
        device = torch.device("cpu")
        print("Model training will run on CPU.")

    else:
        device = torch.device("cpu")
        print("Your answer is invalid. By default, this model will train on CPU. \nYou can terminate this process by pressing Ctrl-Z at any time.")


# Freeze parameters so model won't backprop through them
for param in model.parameters():
    param.requires_grad = False


# Define the classifier
if chosen_arc == 'vgg19':
    hidden_size = int(input("\nYou chose VGG-19. The number of features in VGG-19 is 25088.\nPlease enter the number of hidden units of your choice:  "))
    input_size = 25088
    classifier = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_size, 102),
                                 nn.LogSoftmax(dim=1))

elif chosen_arc == 'densenet121':
    hidden_size = int(input("\nYou chose DenseNet-121. The number of features in DenseNet-121 is 1024.\nPlease enter the number of hidden units of your choice:  "))
    input_size = 1024
    classifier = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_size, 102),
                                 nn.LogSoftmax(dim=1))

model.classifier = classifier
print("\nHere is your model classifier:\n", model.classifier, "\n")


# Define the Loss Function
criterion = nn.CrossEntropyLoss()


# Define the Optimizer and set it to only train the classifier parameters, feature parameters are frozen
learning_rate = float(input("\nThis model will be trained using Cross Entropy Loss function and SGD optimizer. \nPlease enter the learning rate of your choice:  "))
optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)

model.to(device)


# Choose epochs
epoch = int(input("Please choose the number of epochs:  "))


# Training data
with active_session():
    epochs = epoch
    steps = 0
    print_every = 5

    start_time = time.time()
    print("\nTraining begins.\n")
    for e in range(epochs):
        for images, labels in trainloaders:
            steps += 1
            running_loss = 0

            # Move images and labels tensors to the default device
            images, labels = images.to(device), labels.to(device)

            # Reset gradient for optimizer
            optimizer.zero_grad()

            # Feedforward process
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0

                # Turn off dropout for validation loop
                model.eval()

                # Turn off gradient
                with torch.no_grad():

                    # Move images and labels tensors to the default device
                    images, labels = images.to(device), labels.to(device)

                    for images, labels in validloaders:
                        images, labels = images.to(device), labels.to(device)
                        logps = model(images)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                # Turn back on the dropout for next training epoch
                model.train()

                print(f"Epoch {e+1}/{epochs}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Validation loss: {valid_loss/len(validloaders):.3f}.. "
                        f"Validation accuracy: {accuracy/len(validloaders):.3f}")

    total_time = time.time() - start_time
    print("\nTotal time: {:.0f}mins {:.0f}secs".format(total_time // 60, total_time % 60))

# Testing trained model
accuracy = 0

# Turn off dropout for validation loop
model.eval()

# Turn off gradient
with torch.no_grad():

    for images, labels in testloaders:
        images, labels = images.to(device), labels.to(device)
        logps = model(images)

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

# Turn back on the dropout
model.train()

print(f"Test accuracy for your current model: {accuracy/len(testloaders):.3f}")

# Set up model.class_to_idx
model.class_to_idx = image_datasets[0].class_to_idx

# Save the checkpoint
checkpoint = {'input_size': input_size,
              'output_size': 102,
              'arch': chosen_arc,
              'epochs': epochs,
              'classifier': classifier,
              'learning_rate': learning_rate,
              'class_to_idx': model.class_to_idx,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    return model, optimizer

model, optimizer = load_checkpoint('checkpoint.pth')

print("\nYour model has been saved as: ", model)
print("\nYour optimizer has been saved as: ", optimizer)
