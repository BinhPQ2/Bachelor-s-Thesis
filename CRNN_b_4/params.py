import CRNN_b_4.alphabets
# about data and net
alphabet = CRNN_b_4.alphabets.alphabet
imgH = 32 # the height of the input image to network
imgW = 128 # the width of the input image to network
nh = 256 # size of the lstm hidden state
nc = 1 #number of channel for CRNN model

# hardware
cuda = True # enables cuda
multi_gpu = False # whether to use multi gpu
ngpu = 1 # number of GPUs to use. Do remember to set multi_gpu to True!
workers = 0 # number of data loading workers
