import torch # Pytorch package 
import torch.nn as nn # Neural network module from Pytorch
from torch.nn.utils import weight_norm  # Normalization utility for neural network weights


# Class to remove extra values from the end on the temporal dimension
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__() 
        self.chomp_size = chomp_size

    def forward(self, x):
        # Tensor is [batch_size, channels, sequence_length]
        return x[:, :, :-self.chomp_size].contiguous() # contiguous() ensures that the tensor is stored in a contiguous block of memory.


# Class representing a single temporal block in the network
class TemporalBlock(nn.Module):

    def __init__(self, 
                n_inputs, # Number of input channels
                n_outputs, # Number of output channels
                kernel_size, # Size of the window for the convolution operation
                stride, # Step between windows in the convolution operation
                dilation, # Space between kernel elements in the convolution operation
                padding, # Padding for the convolution operation
                dropout=0.2 # Proportion of neurons to drop during training to prevent overfitting
                ):
        super(TemporalBlock, self).__init__() # runs __init__ method of the parent class (nn.Module) 

        # First convolution 
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolution
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Sequential network for the temporal block
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        # If the number of input channels is not equal to the number of output channels, we need to downsample the input to match the output dimensions.
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None 
        # Final ReLU activation function after adding the residual connection
        self.relu = nn.ReLU()
        # Initialize the weights of the convolutional layers using a normal distribution with mean 0 and standard deviation 0.01. 
        self.init_weights()


    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# Class representing the entire Temporal Convolutional Network (TCN)
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
