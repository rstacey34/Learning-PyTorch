import torch 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# print(torch.__version__)

### intro to tensors

### creating tensors

# scalar tensor
scalar = torch.tensor(7)
print(scalar)

## scalar has no dimensions its just a single number
print(scalar.ndim)

#gets tensor back as python int
print(scalar.item())

#vector
vector = torch.tensor([7,7])
print(vector)
##this vector has 1 dimension []
print(vector.ndim)
## number of elements in the vector
print(vector.shape)

##matrix
MATRIX = torch.tensor([[7,8],[9,10]])
print(MATRIX)
# MATRIX has 2 dimensions [[]]
print(MATRIX.ndim)
# MATRIX is [2,2]
print(MATRIX.shape)

### TENSOR
TENSOR = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]]])
print(TENSOR)

## TENSOR has 3 dimensions [[[]]]
print(TENSOR.ndim)
### TENSOR is [1,3,3]
print(TENSOR.shape)
print(TENSOR[0][1][0])

#### random tensors
# create a random tensor
random_tensor = torch.rand(3,4)
print(random_tensor)

## create a random tensor with similar shape to an image tensor
#                       # height, width, colour channels. channels = red,green,blue
random_image_size_tensor = torch.rand(size=(224,224,3))
# colour channels can come first and if thats the case the abve line would look like this 
# torch.rand(size=(3,224,224))
print(random_image_size_tensor) 
print(random_image_size_tensor.ndim)
print(random_image_size_tensor.shape)

## create a tensor of zeros
zeros = torch.zeros(size=(3,4))
print(zeros)

## create a tensor of ones
ones = torch.ones(size=(3,4))
print(ones)

## all tensors default datatype is torch.float32
print(ones.dtype)

### creating a range of tensors and tensors-like
tensor_range = torch.arange(start=0,end=10,step=2)
print(tensor_range)

## creating tensors-like makes a tensor of the same shape as the input
tensors_like = torch.zeros_like(input=tensor_range)
print(tensors_like)

### float_32 tensor
float_32_tensor = torch.tensor([3.0,6.0,9.0],
                               dtype=None, # what dataype is tensor (eg.float_32 or float_16), defualt is float_32 even with none
                               device=None, # can be "cpu" or "cuda" can get errors  with operations on tensors on different devices
                               requires_grad=False) # weater or not to track gradients with this tensors operations
print(float_32_tensor)

##type casting float_32_tensor to float_16 dtype
float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor)

int_32_tensor = torch.tensor([3,6,9],dtype=torch.int32)
print(int_32_tensor)

## works but if throws datatype error then check 
print(float_32_tensor*int_32_tensor)

print(int_32_tensor.device)

## tensor atttributes
print(f"datatype of tensor: {int_32_tensor.dtype}")
print(f"shape of tensor: {int_32_tensor.shape}")
print(f"device of tensor: {int_32_tensor.device}")

## manipulating tensors (tensor operations)

##addition
tensor = torch.tensor([[1,2,3],
                       [4,5,6],
                       [7,8,9]])
print(tensor+10) #tensor([11, 12, 13])

##multiplication
print(tensor*10) #tensor([10, 20, 30])

##subtract
print(tensor-10) #tensor([-9, -8, -7])

print(torch.mul(tensor,10)) #tensor([10, 20, 30]) same as tensor*10

##element wise multiplication
print(tensor*tensor)

## matrix multiplication (dot_product)
print(torch.matmul(tensor,tensor))
## @ symbol is matrix multipliction like matmul
print(tensor @ tensor)
## torch.mm is the same as matmul its an alias for writing less code
print(torch.mm(tensor,tensor))

## tensor transopse
tensor_transpose = torch.rand(3,2)
print(tensor_transpose)
print(tensor_transpose.T) #k .T is the transpose of the tensor

## tensor aggregation: min,max,mean,sum

tensor_agg = torch.arange(0,500,20)
print(tensor_agg)
print(torch.min(tensor_agg),tensor_agg.min)
print(torch.max(tensor_agg),tensor_agg.max)
# print(torch.mean(tensor_agg),tensor_agg.mean) # this line does not work beucase datatype is of type long
print(torch.mean(tensor_agg.type(torch.float32)),tensor_agg.type(torch.float32).mean) # tensor.mean function neds a dtype of float32
print(torch.sum(tensor_agg),tensor_agg.sum)
## postional index of max and min
print(torch.argmax(tensor_agg))
print(torch.argmin(tensor_agg))

#### reshaping, stacking, squeezing and unsqueezing
x_tensor = torch.arange(1.,10.)
print(x_tensor,x_tensor.shape)

##reshaping
x_reshaped = x_tensor.reshape(1,9)
print(x_reshaped,x_reshaped.shape)
x_reshaped = x_tensor.reshape(9,1)
print(x_reshaped,x_reshaped.shape)

##views
z_tensor = x_tensor.view(1,9)
print(z_tensor,z_tensor.shape)

#changing z changes x because a view of a tensor  shares the same memory as the original tensor
z_tensor[:,0]=10
print(z_tensor,"\n",x_tensor)

## stack tensors on top of each other
x_stacked = torch.stack([x_tensor,x_tensor,x_tensor,x_tensor])#dim=0 or 1 for this tensor
print(x_stacked)

## squeeze
x_squeezed = x_reshaped.squeeze()
print(x_reshaped)
print(x_squeezed)

##unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(x_unsqueezed)
x_unsqueezed = x_squeezed.unsqueeze(dim=1)
print(x_unsqueezed)

## permute
y_tensor = torch.rand(size=(224,224,3)) #[height,width,colour_channels]
#permute the y_tensor to rearange the axis(or dim) order
y_permute = y_tensor.permute(2,0,1) # shifts axis 0->1, 1->2, 2->0
print(y_tensor.shape)
print(y_permute.shape) # [colour_channels,height,width]

#### tensors to numpy arrays and visa versa
array = np.arange(1.0,10.0)
print(array)
z_tensor = torch.from_numpy(array) 
# default numpy dtype is float64 so the tensor will be float64
print(z_tensor)
# to avert this try this
z_tensor = torch.from_numpy(array).type(torch.float32)
print(z_tensor,z_tensor.dtype)

## tensor to numpy array
tensor = torch.ones(10)
numpy_tensor = tensor.numpy()
#default dtype for tensor is float32 so numpy array will be float32
print(numpy_tensor,numpy_tensor.dtype)

#### reproducability (trying to take the radom out of random)

RANDOM_SEED = 42
## torch.manual_seed works on the next torch.rand() call
torch.manual_seed(RANDOM_SEED)
random_tensor_a = torch.rand(3,4)

torch.manual_seed(RANDOM_SEED)
random_tensor_b = torch.rand(3,4)

print(random_tensor_a)
print(random_tensor_b)
print(random_tensor_a==random_tensor_b)

#### running tensors and pytorch object on GPUS (making faster computations)

#check for gpu acces with pytorch
print(torch.cuda.is_available())

# setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# count GPUS
print(torch.cuda.device_count())

## putting tensors and models on the GPU
tensor = torch.tensor([1,2,3])
print(tensor,tensor.device)
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu)

## moving tensor back to the cpu
tensor_on_cpu = tensor_on_gpu.cpu().numpy()
print(tensor_on_cpu)

#### exercises
tensor = torch.rand(7,7)
print(tensor.shape)
tensor_trans = torch.mm(tensor.T,torch.rand(7,1))
print(tensor_trans)

RANDOM_SEED = 0
# torch.manual_seed(RANDOM_SEED)
tensor = torch.rand(7,7)
print(tensor.shape)
# torch.manual_seed(RANDOM_SEED)
tensor_trans = torch.mm(tensor.T,torch.rand(7,1))
print(tensor_trans)
GPU_SEED =torch.cuda.manual_seed(1234)
gpu_tensor_1 = torch.rand(2,3,device='cuda')
GPU_SEED =torch.cuda.manual_seed(1234)
gpu_tensor_2 = torch.rand(2,3,device='cuda')

print(gpu_tensor_1)
print(gpu_tensor_2)
gpu_tensor_trans = torch.mm(gpu_tensor_1,gpu_tensor_2.T)
print(gpu_tensor_trans)
print("min of gpu_tensor_trans: ",torch.min(gpu_tensor_trans))
print("max of gpu_tensor_trans: ",torch.max(gpu_tensor_trans))
print("index of min in gpu_tensor_trans: ", torch.argmin(gpu_tensor_trans))
print("index of max in gpu_tensor_trans: ", torch.argmax(gpu_tensor_trans))

tensor = torch.rand(1,1,1,10)
print(tensor)
tensor_squezed = tensor.squeeze()
print(tensor_squezed,tensor_squezed.shape)













