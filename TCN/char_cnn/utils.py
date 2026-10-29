import unidecode
import torch # Tensors + GPU
from torch.autograd import Variable
from collections import Counter # Frequency counting of characters
import observations # Used to load datasets PTB and text8
import os 
import pickle


cuda = torch.cuda.is_available()


def data_generator(args):
    # Finds dataset ptb or text8 in the data folder and loads it. If not found, it downloads the dataset.
    file, testfile, valfile = getattr(observations, args.dataset)('data/')
    file_len = len(file) # Length of the training dataset
    valfile_len = len(valfile) # Length of the validation dataset   
    testfile_len = len(testfile) # Length of the test dataset
    # Creates a Corpus (from collections) object that contains the character-level vocabulary and mappings for the dataset.
    corpus = Corpus(file + " " + valfile + " " + testfile) 

    #############################################################
    # Use the following if you want to pickle the loaded data
    #
    # pickle_name = "{0}.corpus".format(args.dataset)
    # if os.path.exists(pickle_name):
    #     corpus = pickle.load(open(pickle_name, 'rb'))
    # else:
    #     corpus = Corpus(file + " " + valfile + " " + testfile)
    #     pickle.dump(corpus, open(pickle_name, 'wb'))
    #############################################################

    return file, file_len, valfile, valfile_len, testfile, testfile_len, corpus

# Normalize characters to ASCII and return the file content and its length
def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)

# Vocabulary class to handle character-level vocabulary and mappings
class Dictionary(object):
    def __init__(self):
        self.char2idx = {}
        self.idx2char = []
        self.counter = Counter()

    # Add a character to the vocabulary and update its frequency count
    def add_word(self, char):
        self.counter[char] += 1

    # Prepare the character-to-index and index-to-character mappings based on the frequency of characters in the dataset
    def prep_dict(self):
        for char in self.counter:
            if char not in self.char2idx:
                self.idx2char.append(char)
                self.char2idx[char] = len(self.idx2char) - 1

    # Return the size of the vocabulary (number of unique characters)
    def __len__(self):
        return len(self.idx2char)

# Corpus class to handle the character-level vocabulary and mappings for a given dataset
class Corpus(object):
    def __init__(self, string):
        self.dict = Dictionary()
        for c in string:
            self.dict.add_word(c)
        self.dict.prep_dict()

# Convert the input string into a tensor of character indices based on the vocabulary
def char_tensor(corpus, string):
    tensor = torch.zeros(len(string)).long()
    for i in range(len(string)):
        tensor[i] = corpus.dict.char2idx[string[i]]
    return Variable(tensor).cuda() if cuda else Variable(tensor)

# Convert long sequence of character indices into a tensor of shape [batch_size, seq_length] for training
def batchify(data, batch_size, args):
    """The output should have size [L x batch_size], where L could be a long sequence length"""
    # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1)
    if args.cuda:
        data = data.cuda()
    return data

# Extract a batch of input and target sequences from the source data for training or evaluation
def get_batch(source, start_index, args):
    seq_len = min(args.seq_len, source.size(1) - 1 - start_index)
    end_index = start_index + seq_len
    inp = source[:, start_index:end_index].contiguous()
    target = source[:, start_index+1:end_index+1].contiguous()  # The successors of the inp.
    return inp, target

# Save the trained model to a file
def save(model):
    save_filename = 'model.pt'
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)


