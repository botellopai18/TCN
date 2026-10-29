from torch import nn # Neural Networks module from Pytorch
import sys
sys.path.append("../../")
from TCN.tcn import TemporalConvNet # tcn.py from the repo


class TCN(nn.Module):
    # 
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2, emb_dropout=0.2):
        super(TCN, self).__init__() # initialize nn.module
        self.encoder = nn.Embedding(output_size, input_size)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.decoder = nn.Linear(input_size, output_size) # linear layer to map the output of the TCN to the desired output size (number of classes)
        self.decoder.weight = self.encoder.weight
        self.drop = nn.Dropout(emb_dropout)
        self.init_weights()
    # initialize the weights of the embedding and linear layers with uniform distribution and zero bias
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # input has dimension (N, L_in), and emb has dimension (N, L_in, C_in)
        emb = self.drop(self.encoder(x))
        y = self.tcn(emb.transpose(1, 2))
        o = self.decoder(y.transpose(1, 2))
        return o.contiguous()