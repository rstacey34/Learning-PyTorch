import torch 
from torch import nn ## torch nueral networks
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pathlib import Path

"""
Pytorch workflow 

exploring a pytorch end-to-end workflow
1) data (prepare and load)
2) build a model
3) fititng the model to data(training)
4) make predicitions
5) saving and loading a model 
6) putting it all together
"""

#device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# print("cuda version: ",torch._C._cuda_getCompiledVersion())

## data preparing and loading 

## linear regresion formula data to make a straight line with know paramaeters
## formula: Y = a + bX, a = bias, b = weight
weight = 0.7
bias = 0.3

start = 0
end =1
step = 0.02
X = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * X + bias

print(X[:10],y[:10],len(X),len(y))

"""
splitting data into training and test sets
training sets ~60-80% always
validation sets ~10-20% often but not always
testing set ~10-20% always
"""

train_split = int(0.8*len(X))
print(train_split)
X_train,y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:],y[train_split:]
#training set, training labels, testing set, testing labels
print(len(X_train),len(y_train),len(X_test),len(y_test))

def plot_predictions(train_data = X_train,
                     train_labels = y_train,
                     test_data = X_test,
                     test_labels = y_test,
                     predictions=None):
    """
    plots traing data and test dataand compares predictions
    """
    plt.figure(figsize=(10,7))
    #plotting trainign data in blue
    plt.scatter(train_data,train_labels,c="b",s=4, label="Training data")
    
    #plotting test data in green
    plt.scatter(test_data,test_labels,c="g",s=4, label="Testing Data")
  
    if predictions is not None:
        plt.scatter(test_data,predictions,c="r",s=4, label="Predictions")
    plt.legend(prop={"size":4})
    plt.show()
# plot_predictions()


"""
building a pytorch model
"""
##argument in the class parameters is inheritance, so this inherits from nn.Module
class LinearRegressionModel(nn.Module): # almost everything in pytorch inherits from nn.Module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
    ## forward method to define the computation in the model
    def forward(self,x: torch.Tensor)-> torch.Tensor:
        """
        X is expected to be a tensor 
        the "->" show that this fucntion returns a tensor
        """
        return self.weights* x+self.bias # linear regression formula
    
"""
pytorch model building essentials
PyTorch has four (give or take) essential modules you can use to create almost any kind of neural network you can imagine.

They are torch.nn, torch.optim, torch.utils.data.Dataset and torch.utils.data.DataLoader. 
For now, we'll focus on the first two and get to the other two later (though you may be able to guess what they do).
"""

### creating a model from our class
#create a random seed
# torch.manual_seed(42)
# model_0 = LinearRegressionModel()

# print(model_0.state_dict())

# ## making predictions using torch.inference_mode()
# with torch.inference_mode(): ## inference mode gets rid of gradiant tracking and stops from adding grad_fn=<AddBackward0> to the results
#     y_preds = model_0(X_test)
# print(y_preds)

# # plot_predictions(predictions=y_preds)

# """
# training a model from unknown parameters to known parameters

# in other words from poor representation to better representation of data

# one way to do this is with a loss function
#     loss function may be reffered to as cost function or criterion in other areas

# loss function: is a function to measure how wrong your models predictions are to the ideal outputs, lower is better
# optimizer: takes into account the loss of a mode and adjusts the models prameters 
# """

# ## set up a loss function
# loss_fn = nn.L1Loss()
# ##set up an optimizer (stochastic gradient descent)
# optimizer = torch.optim.SGD(params=model_0.parameters(),
#                                                       lr=0.01)#lr = learning rate = possibly the most important hyperparameter we can set

# """
# building a training loop (and testing loop) in pytorch
# 0) loop through the data
# 1) forward pass (this involves data moving through are models forward functions)
# 2) calculate the loss (compare forward pass predictions to ground truth labels)
# 3) optimizer zero grad
# 4) loss backward - moves backwards through the network to calculate the gradients of each of the parameters of our model with respect to the loss (Back propagation)
# 5) optimizer step - use the optimizer to adjust our models parameters to try and improve loss (gradient descent)

# """
# print(model_0.state_dict())

# # an epoch is n number of loops through the data (this is a hyperparameter because we set it ourselves)
# epochs = 200

# ## tracking values 
# epoch_count =[]
# loss_values = []
# test_loss_values = []

# ### training
# # step 0
# for epoch in range(epochs):
#     ## set the model to training mode
#     model_0.train() # sets all parameters that requires gradients to require gradients (True)

#     #1. forward pass
#     y_pred = model_0(X_train) 

#     #2. calculate the loss
#     loss = loss_fn(y_pred,y_train) # predictions first then data 
#     # print(f"loss: {loss}")
#     # 3. optimizer zero grad
#     optimizer.zero_grad()

#     #4. perform back propagation on the loss with respect to the parameters of the model
#     loss.backward()

#     #5. step the optimizer (perform gradient descent)
#     optimizer.step() # by default how te optimizer changes will accumulate through the loop  so we have to sero them above in step 3 for the loop

#     ### testing
#     model_0.eval()#turns off gradient tracking

#     with torch.inference_mode():
#         # 1.forward pass
#         test_predictions = model_0(X_test)

#         #2. calculate the loss
#         test_loss = loss_fn(test_predictions,y_test)
    
#     if epoch %10 ==0:
#         epoch_count.append(epoch)
#         loss_values.append(loss)
#         test_loss_values.append(test_loss)
#         # print whats happending 
#         print(f"Epoch: {epoch} | MAE Train Loss: {loss}| MAE Test Loss: {test_loss}")

#         print(model_0.state_dict())
# print(model_0.state_dict())

# with torch.inference_mode():
#     y_preds_new = model_0(X_test)

# plot_predictions(predictions=y_preds_new)

# ## plot the loss curve
# plt.plot(epoch_count,np.array(torch.tensor(loss_values).numpy()),label="train loss")
# plt.plot(epoch_count,test_loss_values,label ="test loss")
# plt.title("training and test loss curves")
# plt.ylabel("Loss")
# plt.xlabel("Epochs")
# plt.show()

# """
# SAVING AND LOADING A MODEL IN PYTORCH
# 1) torch.save() - allows the saving of a PyTorch object in pythons pickle format
# 2) torch.load() - allows the loading of a saved Pytorch object
# 3) torch.nn.Module.load_state_dict() - this allows you to load a models saved state dictionary

# """

# ## saving a pytorch model
# #1. make a models directory
# MODEL_PATH = Path("models")
# MODEL_PATH.mkdir(parents=True,exist_ok=True)

# #2.create model save path
# MODEL_NAME = "01_pytorch_workflow_model_0.pth"# .pth or .pt is pytorch object extension
# MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# #3. save the model state dict
# print(f"saving model to: {MODEL_SAVE_PATH}")
# torch.save(obj=model_0.state_dict(),
#            f=MODEL_SAVE_PATH)

# #### loading a PyTorch model
# # to load a state dict we have to istantiate a new new instance of our model class
# loaded_model_0 = LinearRegressionModel()

# # load the saved state dict of model_0
# loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
# print(loaded_model_0.state_dict())

# #make some predictions
# loaded_model_0.eval()
# with torch.inference_mode():
#     loaded_model_preds = loaded_model_0(X_test)
# print(y_preds_new == loaded_model_preds)

"""
putting all this together
"""
weight = 0.7
bias = 0.3

start = 0
end =1
step = 0.02
X = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * X + bias

train_split = int(0.8*len(X))
# print(train_split)
X_train,y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:],y[train_split:]
X_train=X_train.to(device)
X_test=X_test.to(device)
y_train=y_train.to(device)
y_test=y_test.to(device)
#training set, training labels, testing set, testing labels
# print(len(X_train),len(y_train),len(X_test),len(y_test))

class LinearRegressionModel2(nn.Module): # almost everything in pytorch inherits from nn.Module
    def __init__(self):
        super().__init__()
        #use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)

    def forward(self,x: torch.Tensor)-> torch.Tensor:
        return self.linear_layer(x)

torch.manual_seed(42)
model_1 = LinearRegressionModel2()
print(model_1.state_dict())

model_1.to(device)

loss_fn = nn.L1Loss()
##set up an optimizer (stochastic gradient descent)
optimizer = torch.optim.SGD(params=model_1.parameters(),
                                                      lr=0.01)#lr = learning rate = possibly the most important hyperparameter we can set
torch.manual_seed(42)

epochs = 200

## tracking values 
epoch_count =[]
loss_values = []
test_loss_values = []

### training
# step 0
for epoch in range(epochs):
    ## set the model to training mode
    model_1.train() # sets all parameters that requires gradients to require gradients (True)

    #1. forward pass
    y_pred = model_1(X_train) 

    #2. calculate the loss
    loss = loss_fn(y_pred,y_train) # predictions first then data 
    # print(f"loss: {loss}")
    # 3. optimizer zero grad
    optimizer.zero_grad()

    #4. perform back propagation on the loss with respect to the parameters of the model
    loss.backward()

    #5. step the optimizer (perform gradient descent)
    optimizer.step() # by default how te optimizer changes will accumulate through the loop  so we have to sero them above in step 3 for the loop

    ### testing
    model_1.eval()#turns off gradient tracking

    with torch.inference_mode():
        # 1.forward pass
        test_predictions = model_1(X_test)

        #2. calculate the loss
        test_loss = loss_fn(test_predictions,y_test)
    
    if epoch %10 ==0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        # print whats happending 
        print(f"Epoch: {epoch} | MAE Train Loss: {loss}| MAE Test Loss: {test_loss}")

        print(model_1.state_dict())
print(model_1.state_dict())

with torch.inference_mode():
    y_preds_new = model_1(X_test)

plot_predictions(predictions=y_preds_new.cpu())

## saving a pytorch model
#1. make a models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,exist_ok=True)

#2.create model save path
MODEL_NAME = "01_pytorch_workflow_model_1.pth"# .pth or .pt is pytorch object extension
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#3. save the model state dict
print(f"saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(),
           f=MODEL_SAVE_PATH)

#### loading a PyTorch model
# to load a state dict we have to istantiate a new new instance of our model class
loaded_model_1 = LinearRegressionModel2()
loaded_model_1.to(device)

# load the saved state dict of model_1
loaded_model_1.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
print(loaded_model_1.state_dict())

#make some predictions
loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_1(X_test)
print(y_preds_new == loaded_model_preds)
plot_predictions(predictions=loaded_model_preds.cpu())

