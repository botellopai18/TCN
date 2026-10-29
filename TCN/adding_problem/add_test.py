import torch
import argparse # Library ommand-line argument parsing
import torch.optim as optim # Contains various optimization algorithms for training neural networks such as Adam, SGD, RMSprop, etc.
import torch.nn.functional as F # Contains various functions for building neural networks, including activation functions, loss functions, etc.
import sys
sys.path.append("../../")
from TCN.adding_problem.model import TCN 
from TCN.adding_problem.utils import data_generator

# 
parser = argparse.ArgumentParser(description='Sequence Modeling - The Adding Problem')
# --batch_size: Specifies the number of samples that will be propagated through the network at once. 
# Default is 32
parser.add_argument('--batch_size', type=int, default=32, metavar='N', 
                    help='batch size (default: 32)')
# --cuda: Specifies whether to use CUDA (GPU) for training. 
# Default is True, meaning that if a CUDA-capable GPU is available, it will be used for training.
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
# --dropout: Specifies the dropout rate, which is a regularization technique used to prevent overfitting in neural networks.
# Default is 0.0, meaning no dropout will be applied.
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
# --clip: Specifies the maximum value for gradient clipping, which is a technique used to prevent exploding gradients during training.
# Default is -1, meaning no gradient clipping will be applied.
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
# --epochs: Specifies the number of epochs for training the model.
# Default is 10
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit (default: 10)')
# --ksize: Specifies the kernel size for the convolutional layers.
# Default is 7
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
# --levels: Specifies the number of levels in the Temporal Convolutional Network (TCN).
# Default is 8
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
# --seq_len: Specifies the length of the input sequences for the adding problem.
# Default is 400
parser.add_argument('--seq_len', type=int, default=400,
                    help='sequence length (default: 400)')
# --log-interval: Specifies the interval at which to report training progress.
# Default is 100, meaning that training progress will be reported every 100 batches.
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100)')
# --lr: Specifies the initial learning rate for the optimizer.
# Default is 4e-3
parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 4e-3)')
# --optim: Specifies the optimization algorithm to use for training the model.
# Default is 'Adam', which is a popular optimization algorithm 
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
# --nhid: Specifies the number of hidden units per layer in the TCN.
# Default is 30
parser.add_argument('--nhid', type=int, default=30,
                    help='number of hidden units per layer (default: 30)')
# --seed: Specifies the random seed for reproducibility of results.
# Default is 1111
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
# parse_args() method is called to parse the command-line arguments and store them in the args variable.
args = parser.parse_args()
# set the random seed for PyTorch to ensure reproducibility of results. 
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Adding problem arguments
# The input to the model consists of two channels: one for the random numbers and one for the binary mask 
input_channels = 2 
# We want a regression output, so the number of classes is set to 1.
n_classes = 1
batch_size = args.batch_size
seq_length = args.seq_len
epochs = args.epochs

print(args)
print("Producing data...")
X_train, Y_train = data_generator(50000, seq_length) # 50000 training samples
X_test, Y_test = data_generator(1000, seq_length) # 1000 test samples


# Note: We use a very simple setting here (assuming all levels have the same # of channels.
channel_sizes = [args.nhid]*args.levels
kernel_size = args.ksize
dropout = args.dropout
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)

if args.cuda:
    model.cuda()
    X_train = X_train.cuda()
    Y_train = Y_train.cuda()
    X_test = X_test.cuda()
    Y_test = Y_test.cuda()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def train(epoch):
    global lr
    model.train()
    batch_idx = 1
    total_loss = 0
    for i in range(0, X_train.size(0), batch_size):
        if i + batch_size > X_train.size(0):
            x, y = X_train[i:], Y_train[i:]
        else:
            x, y = X_train[i:(i+batch_size)], Y_train[i:(i+batch_size)]
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        batch_idx += 1
        total_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            processed = min(i+batch_size, X_train.size(0))
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, X_train.size(0), 100.*processed/X_train.size(0), lr, cur_loss))
            total_loss = 0


def evaluate():
    model.eval()
    with torch.no_grad():
        output = model(X_test)
        test_loss = F.mse_loss(output, Y_test)
        print('\nTest set: Average loss: {:.6f}\n'.format(test_loss.item()))
        return test_loss.item()


for ep in range(1, epochs+1):
    train(ep)
    tloss = evaluate()



