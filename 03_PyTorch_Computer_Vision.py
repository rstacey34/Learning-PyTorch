import torch 
import torchmetrics
from torchmetrics import Accuracy
from torchmetrics import ConfusionMatrix
from torch import nn ## torch nueral networks
import random
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_circles,make_blobs
from sklearn.model_selection import train_test_split
import requests
from pathlib import Path
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from helper_functions import accuracy_fn
from timeit import default_timer as timer
from tqdm.auto import tqdm 
import mlxtend
from mlxtend.plotting import plot_confusion_matrix
device = 'cuda' if torch.cuda.is_available() else 'cpu'


"""
**image data input and output shapes**
[batch_size, width,height,colour_channels]
input examples:
shape = [None, 224, 224, 3] if colour channels = 1 then its greyscale
or 
shape = [32, 224, 224, 3]

output example:
shape =[n] n = number or classes

PyTorch can sometimes default to having colour_channels be higher in the shape order
example:
[batch_size, colour_chanel, height, width] [NCHW]
"""

###########################################################################

"""
ARCHITECHTURE OF A CNN
- input image(s)
- input layer
- convoloution layer
- hidden activation/non-linear activation
- pooling layer
- output layer/linear layer
- output activation
"""
##########################################################################


"""
getting a dataset

MNIST - written numbers dataset
Fashion MNIST - greyscale clothes dataset

"""

# get training data
train_data = datasets.FashionMNIST(root='data',
                                   train=True,
                                   download=True,
                                   transform = ToTensor(),
                                   target_transform=None
                                   )

test_data = datasets.FashionMNIST(root='data',
                                   train=False,
                                   download=True,
                                   transform = ToTensor(),
                                   target_transform=None
                                   )

print(len(train_data),len(test_data))

## see the first training example
print(train_data[0])
print(test_data[0])
image, label = train_data[0]
#prints all class names
class_names = train_data.classes
print(class_names)

#makes a dictionary of class and their index
class_to_idx = train_data.class_to_idx
print(class_to_idx)

#check the shaope
print(f'image shape: {image.shape} -> [colour_channels,height,width]')
print(f'image label: {class_names[label]}')

## visualing the data

# plt.imshow(image) # matplot lib doesn't like this as the colour channel causes errors
# plt.imshow(image.squeeze())
# plt.title(label)
# plt.show()

# plt.imshow(image.squeeze(),cmap='gray')
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()

torch.manual_seed(42)
# fig = plt.figure(figsize=(9,9))
# rows,cols = 4,4
# for i in range(1,rows*cols+1):
#     random_idx = torch.randint(0,len(train_data),size=[1]).item()
#     print(random_idx)
#     img,label = train_data[random_idx]
#     fig.add_subplot(rows,cols,i)
#     plt.imshow(img.squeeze(),cmap='gray')#
#     plt.title(class_names[label])
#     plt.axis(False)

# plt.show()

"""
prepare DataLoader

DataLoader turns our dataset into a Python iterable

more specifically we want to turn our data into batches (or minibatches)

why?
- Because it's more computationally efficient.
- In an ideal world you could do the forward pass and backward pass across all of your data at once.
- But once you start using really large datasets, unless you've got infinite computing power, it's easier to break them up into batches.
- It also gives your model more opportunities to improve.
- With mini-batches (small portions of the data), gradient descent is performed more often per epoch (once per mini-batch rather than once per epoch).
""" 

print(train_data)
print(test_data)
BATCH_SIZE = 32
## DataLoader has a arg called num_workers which is defualt to 0 this stands for the number of cores on the machine to load the data (might be useful to know)
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)#easier to evaluate models when test data is not shuffled

print(train_dataloader)
print(test_dataloader)

## check out what we have created

print(f'DataLoaders: {train_dataloader,test_dataloader}')
print(f'length of the train_dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}')
print(f'length of the test_dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}')

## check out whats inside the training dataloader
train_features_batch,train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape,train_labels_batch.shape)
## show a sample 
torch.manual_seed(42)
# rand_idx = torch.randint(0,len(train_features_batch),size=[1]).item()
# img, label = train_features_batch[rand_idx],train_labels_batch[rand_idx]
# plt.imshow(img.squeeze(),cmap='gray')
# plt.title(class_names[label])
# plt.axis(False)
# print(f'Image size: {img.shape}')
# print(f'label: {label}, label size: {label.shape}')
# plt.show()

"""
building a baseline computer vision model

a baseline model is a simple model you will try and improve upon with subsequent models/experiments
- in short start simple, add complexity later
"""

## create a flatten model

flatten_model = nn.Flatten()

# get a single sample 
x= train_features_batch[0] #shape is ([1,28,28])

# flatten the sample 
output = flatten_model(x) # perform forward pass

print(f'shape before flattening = {x.shape} -> [colour_channels, height,width]')
print(f'shape after flattening = {output.shape} ->[colour_channels, height*width]')

class FasionMNISTModelv0(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units:int,
                 output_shape:int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), #  compress the image into a singular vector
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape)
        )
    def forward(self,x):
        return self.layer_stack(x)


torch.manual_seed(42)
model_0 = FasionMNISTModelv0(input_shape=784, #28*28
                             hidden_units=10,
                             output_shape = len(class_names)
                             ).to('cpu')
print(model_0)
dummy_x = torch.rand([1,1,28,28])
print(model_0(dummy_x))

### set up loss, optimizer and evaluation metrics
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

## create a function to time our experiments
def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    """
    Prints difference between start and end time
    
    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    
    """
    total_time = end-start
    print(f'Training time on {device}: {total_time:.3f} seconds')
    return total_time

start_time = timer()
## usually more code between here
end_time = timer()
print(print_train_time(start=start_time,end=end_time,device='cpu'))

"""
creating a training loop and traing a model on batches of data
note: the optimizer will update the models parameters once per battch rather than once per epoch

- loop through epochs
- loop through training batches, perform training steps, calculate the train loss
- loop through testing batches and perform testing steps, calculate the test loss per batch
- print out  whats happening
- time it all 
"""

torch.manual_seed(42)
torch.cuda.manual_seed(42)
train_time_start_on_cpu = timer()

epochs = 3

for epoch in tqdm(range(epochs)):
    print(f'Epoch: {epoch}\n------------')

    train_loss = 0

    for batch, (X,y) in enumerate(train_dataloader):
        
        model_0.train()
        # do the forward pass
        y_pred = model_0(X)

        ## calculate the loss
        loss = loss_fn(y_pred,y)
        train_loss += loss # accumulate the train loss

        #optimimzer zero grad
        optimizer.zero_grad()

        # loss backward
        loss.backward()

        # optimizer step
        optimizer.step()

        # print out whats happening
        if batch % 400 ==0:
            print(f'Looked at {batch*len(X)}/ {len(train_dataloader.dataset)} sample.')

    # divide total train loss by length of train data loader
    train_loss /= len(train_dataloader)

    ## testing
    test_loss,test_acc = 0,0

    model_0.eval()
    with torch.inference_mode():
        for X_test,y_test in test_dataloader:
            ## forward pass
            test_pred = model_0(X_test)

            ## calculate the loss
            test_loss += loss_fn(test_pred, y_test)

            ## calculate the accuracy 
            test_acc += accuracy_fn(y_true=y_test,
                                    y_pred=test_pred.argmax(dim=1)
                                    )

        ## calculate the test loss average per batch
        test_loss /=len(test_dataloader)

        ## calculate the test acc per batch 
        test_acc /=len(test_dataloader)
    
    ## print out whats happening 
    print(f'\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}')

#calculate the train time
train_time_end_on_cpu = timer()

total_train_time_model_0_cpu = print_train_time(start=train_time_start_on_cpu,
                       end=train_time_end_on_cpu,
                       device=str(next(model_0.parameters()).device))

print(total_train_time_model_0_cpu)

"""
writing an evaluation function to make predicitons and get model 0 results
"""

torch.manual_seed(42)
def eval_model(model:torch.nn.Module,
               data_loader: DataLoader,
               loss_fn: torch.nn.Module,
               accuraccy_fn):
    """
    Returns a dictionary containing the results of model predicting on data_loader
    """
    loss,acc=0,0
    with torch.inference_mode():
        for X,y in data_loader:

            ## make predictions
            y_pred = model(X)

            ## accumulate the loss and accuracy loss per batch
            loss+= loss_fn(y_pred,y)

            acc+=accuraccy_fn(y_true =y,
                              y_pred=y_pred.argmax(dim=1))
            
        ## scale the loss and acc to find the average loss per batch
        loss/=len(data_loader)
        acc/=len(data_loader)

    return {'model_name': model.__class__.__name__,
            'model_loss': loss.item(),
            'model_acc': acc}

##calculate model 0 results on test database

model_0_results = eval_model(model=model_0,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuraccy_fn=accuracy_fn)
print(model_0_results)


## device agnostic code already set up at top of script

#building a better model with non linearity
class FashionMNISTModelV1(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                    out_features=output_shape),
            nn.ReLU()
        )
    def forward(self,x):
        return self.layer_stack(x)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
model_1 = FashionMNISTModelV1(input_shape=784,
                              hidden_units=10,
                              output_shape=len(class_names)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.1)

"""
make a function for training and testing loops
"""

def train_step(model:nn.Module,
               data_loader: DataLoader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               accuracy_fn,
               device:torch.device = device):
    """
    performs a training step with model tryong to learn on data_loader
    """
    train_loss, train_acc = 0,0
    
    # put model into training model
    model.train()

    for batch, (X,y) in enumerate(data_loader):
        X = X.to(device)
        y = y.to(device)
        
        # do the forward pass
        y_pred = model(X)

        ## calculate the loss and accuracy (per batch)
        loss = loss_fn(y_pred,y)
        train_loss += loss # accumulate the train loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) #from logits to prediction labels
        #optimimzer zero grad
        optimizer.zero_grad()

        # loss backward
        loss.backward()

        # optimizer step
        optimizer.step()

        
    # divide total train loss and train accuracy by length of train data loader
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    print(f'Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%\n')

def test_step(model:nn.Module,
               data_loader: DataLoader,
               loss_fn:torch.nn.Module,
               accuracy_fn,
               device:torch.device = device):
    
    test_loss,test_acc = 0,0

    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:

            X, y = X.to(device), y.to(device)
            ## forward pass (outputs raw logits)
            test_pred = model(X)

            ## calculate the loss
            test_loss += loss_fn(test_pred, y)

            ## calculate the accuracy 
            test_acc += accuracy_fn(y_true=y,
                                    y_pred=test_pred.argmax(dim=1)
                                    )

        ## calculate the test loss average per batch
        test_loss /=len(data_loader)

        ## calculate the test acc per batch 
        test_acc /=len(data_loader)
        print(f'Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n')

"""
using our test_step and train_step functions on model_1

"""
torch.manual_seed(42)
torch.cuda.manual_seed(42)

train_time_start_on_gpu = timer()
epochs = 3

for epoch in tqdm(range(epochs)):
    print(f'Epoch: {epoch}\n----------------------')
    train_step(model=model_1,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)
    test_step(model=model_1,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)
    
train_time_end_on_gpu = timer()
total_train_time_model_1_gpu = print_train_time(start=train_time_start_on_gpu,
                                                end=train_time_end_on_gpu,
                                                device=device)
print(total_train_time_model_1_gpu)

"""
## get model 1 results dictionary
model_1_results = eval_model(model=model_1,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuraccy_fn=accuracy_fn)
^^^^^^^^^^^^^ this will error out as the model and data are on different devices after training

remaking the eval_model function
\/\/\/\/\/\/\/
"""


torch.manual_seed(42)
def eval_model(model:torch.nn.Module,
               data_loader: DataLoader,
               loss_fn: torch.nn.Module,
               accuraccy_fn,
               device=device):
    """
    Returns a dictionary containing the results of model predicting on data_loader
    """
    loss,acc=0,0
    with torch.inference_mode():
        for X,y in data_loader:
            #make data device agnostic
            X,y = X.to(device),y.to(device)

            ## make predictions
            y_pred = model(X)

            ## accumulate the loss and accuracy loss per batch
            loss+= loss_fn(y_pred,y)

            acc+=accuraccy_fn(y_true =y,
                              y_pred=y_pred.argmax(dim=1))
            
        ## scale the loss and acc to find the average loss per batch
        loss/=len(data_loader)
        acc/=len(data_loader)

    return {'model_name': model.__class__.__name__,
            'model_loss': loss.item(),
            'model_acc': acc}

model_1_results = eval_model(model=model_1,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuraccy_fn=accuracy_fn,
                             device=device)

print(model_1_results)

"""
model 2 a convolutional nueral network (CNN)


"""

# class FasionMNISTModelV2(nn.Module):
#     def __init__(self, input_shape: int,
#                  hidden_units: int,
#                  output_shape: int):
#         super().__init__()
#         self.conv_block_1 = nn.Sequential(
#             nn.Conv2d(in_channels=input_shape,
#                       out_channels=hidden_units,
#                       kernel_size=3,
#                       stride=1,
#                       padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=hidden_units,
#                       out_channels=hidden_units,
#                       kernel_size=3,
#                       stride=1,
#                       padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,
#                          stride=2)
#         ),
#         self.conv_block_2 = nn.Sequential(
#             nn.Conv2d(in_channels=hidden_units,
#                       out_channels=hidden_units,
#                       kernel_size=3,
#                       stride=1,
#                       padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=hidden_units,
#                       out_channels=hidden_units,
#                       kernel_size=3,
#                       stride=1,
#                       padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         ),
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(in_features=int(hidden_units*7*7), # theres a trick to calculating this 
#                       out_features=output_shape)
#         )
#     def forward(self,x):
#         x = self.conv_block_1(x)
#         print(f'Inside model after first block: {x.shape}')
#         x = self.conv_block_2(x)
#         print(f'Inside model after second block: {x.shape}')
#         x = self.classifier(x)
#         print(f'Inside model after classifier block: {x.shape}')
#         return x


# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# model_2 = FasionMNISTModelV2(input_shape=1,
#                              hidden_units=10,
#                              output_shape=len(class_names)).to(device)

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# # images = torch.randn(size=(32,3,64,64))
# # test_image = images[0]
# # print(f'Image batch shape{images.shape}')
# # print(f'single image shape: {test_image.shape}')
# # print(f'test image:\n {test_image}')

# # conv_layer = nn.Conv2d(in_channels=3,
# #                        out_channels=10,
# #                        kernel_size=(3,3),
# #                        stride=1,
# #                        padding=0)

# # conv_output = conv_layer(test_image)
# # print(conv_output)
# # print(conv_output.shape)

# # max_pool_layer = nn.MaxPool2d(kernel_size=2)
# # test_image_through_conva_and_max_pool = max_pool_layer(conv_output)
# # print(f'test image shape after conv and maxpool: {test_image_through_conva_and_max_pool.shape}')

# torch.manual_seed(42)

# ## set up loss function and optimizer for model 2

# loss_fn = nn.CrossEntropyLoss()

# optimizer = torch.optim.SGD(params=model_2.parameters(),
#                             lr=0.1)

# ## training and testing model 2 using the functions

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# train_time_start_model_2_on_gpu = timer()

# epoch = 3

# for epoch in tqdm(range(epochs)):
#     print(f'Epoch: {epoch}\n-------------------------')

#     train_step(model=model_2,
#                data_loader=train_dataloader,
#                loss_fn=loss_fn,
#                optimizer=optimizer,
#                accuracy_fn=accuracy_fn,
#                device=device)
    
#     test_step(model=model_2,
#               data_loader=test_dataloader,
#               loss_fn=loss_fn,
#               accuracy_fn=accuracy_fn,
#               device=device)

# train_time_end_model_2_on_gpu = timer()
# total_train_time_model_2_gpu = print_train_time(start=train_time_start_model_2_on_gpu,
#                                                 end=train_time_end_model_2_on_gpu,
#                                                 device=device)
# print(total_train_time_model_2_gpu)

class FashionMNISTModelV2(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*7*7, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=1, 
    hidden_units=10, 
    output_shape=len(class_names)).to(device)

torch.manual_seed(42)

# Create sample batch of random numbers with same size as image batch
images = torch.randn(size=(32, 3, 64, 64)) # [batch_size, color_channels, height, width]
test_image = images[0] # get a single image for testing
print(f"Image batch shape: {images.shape} -> [batch_size, color_channels, height, width]")
print(f"Single image shape: {test_image.shape} -> [color_channels, height, width]") 
print(f"Single image pixel values:\n{test_image}")

torch.manual_seed(42)

# Create a convolutional layer with same dimensions as TinyVGG 
# (try changing any of the parameters and see what happens)
conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=10,
                       kernel_size=3,
                       stride=1,
                       padding=0) # also try using "valid" or "same" here 

# Pass the data through the convolutional layer
conv_layer(test_image) # Note: If running PyTorch <1.11.0, this will error because of shape issues (nn.Conv.2d() expects a 4d tensor as input)

# Add extra dimension to test image
print(test_image.unsqueeze(dim=0).shape)

# Pass test image with extra dimension through conv_layer
print(conv_layer(test_image.unsqueeze(dim=0)).shape)

torch.manual_seed(42)
# Create a new conv_layer with different values (try setting these to whatever you like)
conv_layer_2 = nn.Conv2d(in_channels=3, # same number of color channels as our input image
                         out_channels=10,
                         kernel_size=(5, 5), # kernel is usually a square so a tuple also works
                         stride=2,
                         padding=0)

# Pass single image through new conv_layer_2 (this calls nn.Conv2d()'s forward() method on the input)
print(conv_layer_2(test_image.unsqueeze(dim=0)).shape)

# Check out the conv_layer_2 internal parameters
print(conv_layer_2.state_dict())

# Get shapes of weight and bias tensors within conv_layer_2
print(f"conv_layer_2 weight shape: \n{conv_layer_2.weight.shape} -> [out_channels=10, in_channels=3, kernel_size=5, kernel_size=5]")
print(f"\nconv_layer_2 bias shape: \n{conv_layer_2.bias.shape} -> [out_channels=10]")

# Print out original image shape without and with unsqueezed dimension
print(f"Test image original shape: {test_image.shape}")
print(f"Test image with unsqueezed dimension: {test_image.unsqueeze(dim=0).shape}")

# Create a sample nn.MaxPoo2d() layer
max_pool_layer = nn.MaxPool2d(kernel_size=2)

# Pass data through just the conv_layer
test_image_through_conv = conv_layer(test_image.unsqueeze(dim=0))
print(f"Shape after going through conv_layer(): {test_image_through_conv.shape}")

# Pass data through the max pool layer
test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
print(f"Shape after going through conv_layer() and max_pool_layer(): {test_image_through_conv_and_max_pool.shape}")

torch.manual_seed(42)
# Create a random tensor with a similiar number of dimensions to our images
random_tensor = torch.randn(size=(1, 1, 2, 2))
print(f"Random tensor:\n{random_tensor}")
print(f"Random tensor shape: {random_tensor.shape}")

# Create a max pool layer
max_pool_layer = nn.MaxPool2d(kernel_size=2) # see what happens when you change the kernel_size value 

# Pass the random tensor through the max pool layer
max_pool_tensor = max_pool_layer(random_tensor)
print(f"\nMax pool tensor:\n{max_pool_tensor} <- this is the maximum value from random_tensor")
print(f"Max pool tensor shape: {max_pool_tensor.shape}")

# Setup loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), 
                             lr=0.1)

torch.manual_seed(42)

# Measure time
from timeit import default_timer as timer
train_time_start_model_2 = timer()

# Train and test model 
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, 
        model=model_2, 
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device
    )
    test_step(data_loader=test_dataloader,
        model=model_2,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device
    )

train_time_end_model_2 = timer()
total_train_time_model_2 = print_train_time(start=train_time_start_model_2,
                                           end=train_time_end_model_2,
                                           device=device)

print(total_train_time_model_2)

model_2_results = eval_model(model=model_2,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuraccy_fn=accuracy_fn,
                             device=device)

print(model_2_results)

"""
comparing results accros experiments

"""

compare_results = pd.DataFrame([model_0_results,
                                model_1_results,
                                model_2_results])
print(compare_results)

## add training time to results comparison

compare_results['Training_time']=[total_train_time_model_0_cpu,
                                  total_train_time_model_1_gpu,
                                  total_train_time_model_2]
print(compare_results)

## visualize the models results

compare_results.set_index('model_name')['model_acc'].plot(kind='barh')
plt.xlabel('accuracy (%)')
plt.ylabel('model')
plt.show()

## make and evalualte random predictions with the best model

def make_predictions(model: nn.Module,
                     data: list,
                     device: torch.device = device):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            #prepare sample (add a batch dimension and pass to target device)
            sample = torch.unsqueeze(sample,dim=0).to(device)

            # forward pass (model outputs raw logits)
            pred_logit = model(sample)

            #get prediction probability (logit to prediciton probability)
            pred_prob = torch.softmax(pred_logit.squeeze(),dim=0)

            # get pred_prob of gpu f0r further calculations
            pred_probs.append(pred_prob.cpu())
    return torch.stack(pred_probs)              

# random.seed(42)
test_samples = []
test_labels = []

for sample, label in random.sample(list(test_data),k=9):
    test_samples.append(sample)
    test_labels.append(label)

print(f"Test sample image shape: {test_samples[0].shape}\nTest sample label: {test_labels[0]} ({class_names[test_labels[0]]})")

# make predicitons

pred_probs = make_predictions(model=model_2,
                              data=test_samples)

# convert prediciton probabilities into labels

pred_classes = pred_probs.argmax(dim=1)

plt.figure(figsize=(9,9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
    # create subplot
    plt.subplot(nrows,ncols, i+1)

    # plot target image
    plt.imshow(sample.squeeze(), cmap='gray')

    # find the prediction
    pred_label = class_names[pred_classes[i]]

    # the truth label (in text form)
    truth_label = class_names[test_labels[i]]

    title_text = f'Pred: {pred_label} | Truth {truth_label}'
    # check for equality between pred and truth can change colour of title text
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c='g')# green text is true
    else:
        plt.title(title_text, fontsize=10,c='r')# red text is false
    plt.axis(False)        
plt.show()

## making a confusion matrix for futher prediction evaluation

y_preds = []
model_2.eval()
with torch.inference_mode():
    for X,y in tqdm(test_dataloader, desc='Making Predictions...'):
        
        X,y = X.to(device),y.to(device)
        
        y_logit = model_2(X)

        y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)
        
        y_preds.append(y_pred.cpu())

# print(y_preds)
y_pred_tensor = torch.cat(y_preds)
print(y_pred_tensor[:10])

# setup confusion matrix instance and compare predicitons

confmat = ConfusionMatrix(num_classes = len(class_names),task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target = test_data.targets)

print(confmat_tensor)

# plot the confusion matrix

fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(10,7)
)
plt.show()

## save and load our best model 

MODEL_PATH = Path('models')
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)

MODEL_NAME = '03_pytorch_computer_vision_model_2.pth'
MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME

torch.save(obj=model_2.state_dict(),
           f=MODEL_SAVE_PATH)

torch.manual_seed(42)
loaded_model_2 = FashionMNISTModelV2(input_shape=1,
                                     hidden_units=10,
                                     output_shape=len(class_names))

loaded_model_2.load_state_dict(torch.load(MODEL_SAVE_PATH))
loaded_model_2.to(device)

print(model_2_results)

torch.manual_seed(42)
loaded_model_2_results = eval_model(model=loaded_model_2,
                                    data_loader=test_dataloader,
                                    loss_fn=loss_fn,
                                    accuraccy_fn=accuracy_fn,
                                    device=device)
print(loaded_model_2_results)

# check if model results are close to each other

print(torch.isclose(torch.tensor(model_2_results['model_loss']),
              torch.tensor(loaded_model_2_results['model_loss']),
              atol=1e-02))


































