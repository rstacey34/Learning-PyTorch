import torch 
import torchmetrics
from torchmetrics import Accuracy
from torch import nn ## torch nueral networks
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_circles,make_blobs
from sklearn.model_selection import train_test_split
import requests
from pathlib import Path
"""
Architechture of a classification model 

classification is a problem of predicting whether something is one thing or another (there can be multiple things as the options)

"""

## make 1000 samples
n_samples = 1000

## create curcles
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

print(len(X),len(y))

print(f"first five samples of X: \n{X[:5]}")
print(f"first five samples of y: \n{y[:5]}")

### make a dataframe of circle data
circles = pd.DataFrame({"X1":X[:, 0],
                        "X2":X[:, 1],
                        "label":y})
print(circles.head(10))

### visualize,visualize,visualize
plt.scatter(x=X[:,0],
            y=X[:,1],
            c=y,
            cmap=plt.cm.RdYlBu)
# plt.show()

## note: data we are working with here is known as a toy dataset



X_sample = X[0]
y_sample = y[0]
print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
print(f"Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")


"""
turning our data into tensors and creating train and test splits
"""
## turning numpy array into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

print(X[:5],y[:5])

##split into train and test set

X_train,X_test, y_train,y_test = train_test_split(X,
                                                  y,
                                                  test_size=0.2,
                                                  random_state=42)
print(len(X_train),len(X_test),len(y_train),len(y_test))

##build a model to classify blue and red dots

#device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
"""
1. Subclasses nn.Module (almost all PyTorch models are subclasses of nn.Module).
2. Creates 2 nn.Linear layers in the constructor capable of handling the input and output shapes of X and y.
3. Defines a forward() method containing the forward pass computation of the model.
4. Instantiates the model class and sends it to the target device.
"""
## part 1
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2,out_features=5)#takes in 2 features and upscales it to 5 features
        self.layer_2 = nn.Linear(in_features=5,out_features=1)#in_features of second layer has to match the out_features of the previous layer

    def forward(self,x):
        return self.layer_2(self.layer_1(x))#x -> layer_1 -> layer_2 -> output

model_0 = CircleModelV0().to(device)
print(model_0)

## replicating the model above using nn.Sequential()
model_0 = nn.Sequential(nn.Linear(in_features=2,out_features=5),
                        nn.Linear(in_features=5,out_features=1)).to(device)
print(model_0)
print(model_0.state_dict())

## make some predictions with the model
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
print(f"Length of predictions: {len(untrained_preds)}, shape of predictions: {untrained_preds.shape}")
print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
print(f"\nFirst 10 test labels:\n{y_test[:10]}")


"""
picking a loss function and optimizer
And for a binary classification problem (like ours), 
you'll often use binary cross entropy as the loss function.
"""

## loss function
# loss_fn = nn.BCELoss() # BCELoss = no sigmoid built-in
loss_fn = nn.BCEWithLogitsLoss() #This has the sigmoid activation fucntion built in

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

##calculate accuracy - out of 100 examples what percentage does the model get right
def accuracy_fn(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc

"""
training the model
1. forward pass
2. calculate the loss
3. optimizer zero grad
4. loss backward (backpropagation)
5. optimizer step (gradient descent)
"""

#going from raw logits -> prediction probabilities -> predicition labels
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]
print(y_logits)

## use the sigmoid activation fucntion on the logits to turn them into prediction probabilities
y_pred_probs = torch.sigmoid(y_logits)
print(y_pred_probs)
y_preds = torch.round(y_pred_probs)
print(f"predicted labels:\n {y_preds}")
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

print(torch.eq(y_preds.squeeze(),y_pred_labels.squeeze()))

## building a training and testing loop

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 1000

X_train,y_train = X_train.to(device),y_train.to(device)
X_test,y_test = X_test.to(device),y_test.to(device)

# for epoch in range(epochs):
#     model_0.train()

#     # forward pass
#     y_logits = model_0(X_train).squeeze()
#     y_pred = torch.round(torch.sigmoid(y_logits)) # turns logits -> pred probs -> pred labels

#     # calculate loss/accuracy
#     loss = loss_fn(y_logits,
#                    y_train)
#     acc = accuracy_fn(y_true=y_train,
#                       y_pred=y_pred)
    
#     # optizer zero grad
#     optimizer.zero_grad()

#     ## loss backward
#     loss.backward()

#     ##optimizer sttep
#     optimizer.step()

#     ###testing
#     model_0.eval()
#     with torch.inference_mode():
#         #forward pass
#         test_logits = model_0(X_test).squeeze()
#         test_pred = torch.round(torch.sigmoid(test_logits))
        
#         #calculate the teszt loss/accuracy
#         test_loss = loss_fn(test_logits,
#                             y_test)
#         test_acc = accuracy_fn(y_true=y_test,
#                                y_pred=test_pred)
        
#         #print out whats happening
#         if epoch %10 ==0:
#             print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")


"""
make predicitions and evaluate model
at first look model doesnt look like its learning anything 

"""

## download helper functions from PyTorch repo
if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py","wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

# ## plot descision boundary 
# plt.figure(figsize=(12,6))
# plt.subplot(1,2,1)
# plt.title("Train")
# plot_decision_boundary(model_0,X_train,y_train)
# plt.subplot(1,2,2)
# plt.title("Test")
# plot_decision_boundary(model_0,X_test,y_test)
# plt.show()


"""
improving a model (from a model perspective)
- add more layers
- add more hidden units - go from 5 hidden units to 10 hidden units
- fit for longer - more epochs
- change the activation functions
- change the learning rate
- change the loss function
"""

class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2,
                                 out_features=10)
        self.layer_2 = nn.Linear(in_features=10,
                                 out_features=10)
        self.layer_3 = nn.Linear(in_features=10,
                                 out_features=1)
        
    def forward(self,x):
        return self.layer_3(self.layer_2(self.layer_1(x)))

model_1 = CircleModelV1().to(device)
print(model_1)

## create a loss function
loss_fn = nn.BCEWithLogitsLoss()

## create a optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.1)
## write a traing and evaluation loop for model 1
torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 1000

X_train,y_train = X_train.to(device),y_train.to(device)
X_test,y_test = X_test.to(device),y_test.to(device)

# for epoch in range(epochs):
#     model_1.train()

#     y_logits = model_1(X_train).squeeze()

#     y_pred = torch.round(torch.sigmoid(y_logits))

#     loss = loss_fn(y_logits,y_train)

#     acc = accuracy_fn(y_true=y_train,
#                       y_pred=y_pred)
    
#     optimizer.zero_grad()

#     loss.backward()

#     optimizer.step()

#     model_1.eval()
#     with torch.inference_mode():
#         test_logits = model_1(X_test).squeeze()
#         test_pred = torch.round(torch.sigmoid(test_logits))
        
#         #calculate the teszt loss/accuracy
#         test_loss = loss_fn(test_logits,
#                             y_test)
#         test_acc = accuracy_fn(y_true=y_test,
#                                y_pred=test_pred)
        
#         #print out whats happening
#         if epoch %100 ==0:
#             print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

# ## plot descision boundary 
# plt.figure(figsize=(12,6))
# plt.subplot(1,2,1)
# plt.title("Train")
# plot_decision_boundary(model_1,X_train,y_train)
# plt.subplot(1,2,2)
# plt.title("Test")
# plot_decision_boundary(model_1,X_test,y_test)
# plt.show()

## preparing data to see if our model can fit a straight line
weight = 0.7
bias = 0.3

start = 0
end =1
step = 0.01
X_regression = torch.arange(start,end,step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias

## create train and test spliit

train_split = int(0.8*len(X_regression))
X_train_regression,y_train_regression = X_regression[:train_split],y_regression[:train_split]
X_test_regression,y_test_regression = X_regression[train_split:],y_regression[train_split:]

plot_predictions(train_data=X_train_regression,
                 train_labels=y_train_regression,
                 test_data=X_test_regression,
                 test_labels=y_test_regression)
plt.show()

## adjust model_1 to fit a straight line

model_2 = nn.Sequential(
    nn.Linear(in_features=1,out_features=10),
    nn.Linear(in_features=10,out_features=10),
    nn.Linear(in_features=10,out_features=1)
).to(device)

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_2.parameters(),
                            lr=0.01)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 1000

X_train_regression,y_train_regression = X_train_regression.to(device),y_train_regression.to(device)
X_test_regression,y_test_regression = X_test_regression.to(device),y_test_regression.to(device)

for epoch in range(epochs):
    y_pred = model_2(X_train_regression)
    loss= loss_fn(y_pred,y_train_regression)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_2.eval()
    with torch.inference_mode():
        test_pred = model_2(X_test_regression)
        test_loss = loss_fn(test_pred,y_test_regression)

    if epoch % 100 == 0: 
        print(f"Epoch: {epoch} | Train loss: {loss:.5f}, Test loss: {test_loss:.5f}")

model_2.eval()
with torch.inference_mode():
    y_preds = model_2(X_test_regression)
plot_predictions(train_data=X_train_regression.cpu(),
                 train_labels=y_train_regression.cpu(),
                 test_data=X_test_regression.cpu(),
                 test_labels=y_test_regression.cpu(),
                 predictions=y_preds.cpu())
plt.show()

"""
non-linearity

- combinning the linear and non linear functions in a model

"""
n_samples = 1000

X,y=make_circles(n_samples,
                 noise=0.03,
                 random_state=42)

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

## split into tran test split

X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 test_size=0.2,
                                                 random_state=42)

"""
building a model with non linear activation functions
- nn.sigmoid is a non linear function
- nn.ReLu is another non linearclassification function
"""

class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2,
                                out_features=10)
        self.layer_2 =nn.Linear(in_features=10,
                                out_features=10)
        self.layer_3 = nn.Linear(in_features=10,
                                 out_features=1)
        self.relu = nn.ReLU() # ReLu is a non linear activation function

    def forward(self,x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
            
model_3 = CircleModelV2().to(device)
print("model_3\n")
print(model_3.state_dict())

# set up loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_3.parameters(),
                            lr=0.1)


torch.manual_seed(42)
torch.cuda.manual_seed(42)

X_train,y_train = X_train.to(device),y_train.to(device)
X_test,y_test = X_test.to(device),y_test.to(device)


epochs = 1000

for epoch in range(epochs):
    model_3.train()

    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits,y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    ##testing
    model_3.eval()
    with torch.inference_mode():
        test_logits = model_3(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits,y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

        ## print out whats happening
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

model_3.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_3,X_train,y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_3,X_test,y_test)
plt.show()


"""
increasing model_3 accuracy
- replicating non linear activation functions

"""

## straight line data
A = torch.arange(-10,10,1, dtype=torch.float)

def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.tensor(0),x) # inputs mus be tensor
print(relu(A))
plt.plot(relu(A))
plt.show()
def sigmoid(x):
    return 1/(1+torch.exp(-x))
plt.plot(sigmoid(A))
plt.show()

"""
building a multiclass classification model

"""

## create toydata

## set hyperparameters
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

## create multiclass data
X_blob,y_blob = make_blobs(n_samples=1000,
                           n_features=NUM_FEATURES,
                           centers=NUM_CLASSES,
                           cluster_std=1.5, # give clusters a little shake up 
                           random_state=RANDOM_SEED)

##turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

## make train test split
X_blob_train,X_blob_test,y_blob_train,y_blob_test = train_test_split(X_blob,
                                                                     y_blob,
                                                                     test_size=0.2,
                                                                     random_state=RANDOM_SEED)

## plot the data (visualize,visualize,visualize)

plt.figure(figsize=(10,7))
plt.scatter(X_blob[:,0],X_blob[:,1],c=y_blob,cmap=plt.cm.RdYlBu)
plt.show()

## building the multiclass classification model

X_blob_train,X_blob_test = X_blob_train.to(device),X_blob_test.to(device)
y_blob_train,y_blob_test = y_blob_train.to(device),y_blob_test.to(device)


class BlobModel(nn.Module):
    def __init__(self,input_features,output_features,hidden_units=8):
        """
        initializes multiclass classification model
        args:
            input_features (int): number of input features to the model
            output_features (int): number of output features (number of output classes)
            hidden _units (int): number of hidden units between layers defualt = 8
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features,
                      out_features=hidden_units),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=hidden_units),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=output_features)
        )
    def forward(self,x):
        return self.linear_layer_stack(x)    

model_4 = BlobModel(input_features=2,
                    output_features=4,
                    hidden_units=8).to(device)
print(model_4)


## creata loss function and optimizer for multiclass classification 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_4.parameters(),
                            lr=0.1)

"""
getting prediction and probabilities for a multiclass model

in order to evaluate and train and test the model we need to convert the models 
outputs (logits) to prediction probabilities and then to prediction labels.
logits (raw output of the model) -> pred probs (use torch.softmax) -> pred labels ( take the argmax of the prediction probablities)
"""

model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)
print(y_logits[:5])
y_pred_probs = torch.softmax(y_logits,dim=1)
print(y_pred_probs[:5])# prints how much the model thinks its class 0,1,2 or 3 in that order

#converting the models prediction probablities to prediction labels
y_preds = torch.argmax(y_pred_probs,dim=1)
print("y predictions")
print(y_preds)

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
epochs = 100


for epoch in range(epochs):
    
    model_4.train()

    y_logits = model_4(X_blob_train)
    y_pred = torch.softmax(y_logits,dim=1).argmax(dim=1)

    loss = loss_fn(y_logits,y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train,
                      y_pred=y_pred)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    ## test
    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_pred = torch.softmax(test_logits,dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits,y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test,
                               y_pred=test_pred)
    #print whats happenning
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")    
    

## visualize the data
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)
y_pred_probs = torch.softmax(y_logits,dim=1)
print(y_pred_probs[:10])

y_preds = torch.argmax(y_pred_probs,dim=1)
print(y_preds[:10])
print(y_blob_test[:10])

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_4,X_blob_train,y_blob_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_4,X_blob_test,y_blob_test)
plt.show()

"""
relu functions are commented out in the class for testing and the model still works 
feel free to uncomment those 
"""
##########################################################


"""
a few more classification metrics
- accuracy - out of 100 samples, how many does our model get right? - torchmetrics.Accuracy() or sklearn.metrics.accuracy_score()
- precission - Proportion of true positives over total number of samples. Higher precision leads to less false positives (model predicts 1 when it should've been 0) - torchmetrics.Precision() or sklearn.metrics.precision_score()
- F1-score - Combines precision and recall into one metric. 1 is best, 0 is worst. - torchmetrics.F1Score() or sklearn.metrics.f1_score()
- recall - Proportion of true positives over total number of true positives and false negatives (model predicts 0 when it should've been 1). Higher recall leads to less false negatives - torchmetrics.Recall() or sklearn.metrics.recall_score()
- confusion matrix - Compares the predicted values with the true values in a tabular way, if 100% correct, all values in the matrix will be top left to bottom right (diagnol line). - torchmetrics.ConfusionMatrix or sklearn.metrics.plot_confusion_matrix()
- classification report - Collection of some of the main classification metrics such as precision, recall and f1-score. - sklearn.metrics.classification_report()

be aware of precision recall trade off - If you increase precision, it will reduce recall, and vice versa. This is called the precision/recall tradeoff
"""

##  setup metric
torchmetric_accuracy = Accuracy(task='multiclass', num_classes=4).to(device)

##calculate accuracy

print(torchmetric_accuracy(y_preds,y_blob_test))











