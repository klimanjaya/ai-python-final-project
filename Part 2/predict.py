# Imports
import numpy as np
import os, random
import json

import torch
from torchvision import models
from PIL import Image

# Ask for model architecture and load pretrained model
chosen_arc = input("You will load the pretrained model for prediction here. Please note that you must use the same model architecture that you trained and previously saved. Otherwise, this process will terminate due to unmatched model.\n\nPlease enter vgg19 or densenet121:  ")

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


# Load model saved in checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'])

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    return model

model = load_checkpoint('checkpoint.pth')
print("You have successfully uploaded the checkpoint model.\nModel: ", model)

# Load an image from flowers folder
img = random.choice(os.listdir('./flowers/test/3/'))
image_path = './flowers/test/3/' + img

# Define process_image function
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model
    im = Image.open(image_path)

    # Resize image with shortest side 256px keeping aspect ratio
    width, height = im.size
    ratio = width/height
    if ratio > 0:
        height = 256
        im = im.resize((int(ratio*height), height))
    elif (1/ratio) > 0:
        width = 256
        im = im.resize((width, int(width/ratio)))

    # Center crop with PIL (Note to self: orientation of crop begins from top-left)
    width, height = im.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    im = im.crop((left, top, right, bottom))

    # Color channels normalization
    np_im = np.array(im)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    im = (np_im - mean)/std

    # Transpose color channel from 3rd D to 1st D
    im = im.transpose(2,0,1)

    return im

# Define predict function
k = int(input("\nPlease indicate which K value for the top-K prediction you would like to evaluate. Enter 1 or 5:  "))
def predict(image_path, model, topk=k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Implement the code to predict the class from an image file

    # Ask for GPU or CPU
    gpu = input("\nWould you like to train the model on GPU? (Y/N)  ")
    if (gpu in ('Y', 'y')) or (gpu in ('Yes', 'yes')):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU is enabled for prediction.")
        else:
            device = torch.device("cpu")
            print("GPU is not available. Your prediction will run on CPU.")

    elif (gpu in ('N', 'n')) or (gpu in ('No', 'no')):
         device = torch.device("cpu")
         print("Prediction will run on CPU.")

    else:
        gpu1 = input("Your answer is invalid. Would you like to run prediction on GPU? (Y/N)  ")
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


    # Turn off dropout for validation loop
    model.eval()

    # The image
    image = process_image(image_path)

    # Convert numpy to tensor
    image = torch.from_numpy(np.array([image])).float()
    image = image.to(device)

    # Feedforward
    model = model.to(device)
    logps = model(image)


    # Get probabilities and indices for top-K
    ps = torch.exp(logps)
    top_p, top_idx = ps.topk(k, dim=1)

    idx_to_class = {value: key for key, value in model.class_to_idx.items()}

    topk_probs = top_p.tolist()[0]
    topk_idx = top_idx.tolist()[0]

    topk_classes = []
    for i in topk_idx:
        topk_classes.append(idx_to_class[i])

    return topk_probs, topk_classes

# Run prediction function
probs, classes = predict(image_path, model)

# Ask if user wants to print out the top K classes along with their associated probabilities
print_topk = input("\nWould you like to print the top K classes along with their associated probabilities? (Y/N)")
if (print_topk in ('Y', 'y')) or (print_topk in ('Yes', 'yes')):
    print("Top K classes: {} with their associated probablities: {}".format(classes, probs))
elif (print_topk in ('N', 'n')) or (print_topk in ('No', 'no')):
    pass
else:
    print("\nYou have an invalid answer.\nJust in case, here are the top K classes: {} with their associated probabilities: {}". format(classes, probs))

# Label mapping
json_file = input("\nPlease enter the name of JSON file that maps the class values to other category names, for example, class_to_name.json:  ")
with open(json_file, 'r') as f:
    class_to_name = json.load(f)

# Printing actual names and their associated probabilities
topk_names = []
for c in classes:
    topk_names.append(class_to_name[c])

print("\nHere are the top K predicted names for the randomly uploaded image: {}, with their associated probabilities: {}.". format(topk_names, probs))
