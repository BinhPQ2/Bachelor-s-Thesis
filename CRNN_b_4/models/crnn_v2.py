import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2] #kernel size
        ss = [1, 1, 1, 1, 1, 1, 1] #stride size
        ps = [1, 1, 1, 1, 1, 1, 0] #padding size
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential(OrderedDict([
            #0
            ('conv0',nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)), #64x
            ('relu0',nn.ReLU(True)),
            ('pooling0',nn.MaxPool2d(kernel_size=2)),

            #1
            ('conv1', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
            ('relu1',nn.ReLU(True)),
            ('pooling1',nn.MaxPool2d(kernel_size=2)),

            #2
            ('conv2',nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
            ('batchnorm2',nn.BatchNorm2d(256)),
            ('relu2',nn.ReLU(True)),

            ##3
            ('conv3',nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            ('Relu3',nn.ReLU(True)),
            ('pooling2',nn.MaxPool2d(kernel_size=(2,2), stride=(2,1), padding=(0,1))),

            #4
            ('conv4',nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)),
            ('batchnorm4',nn.BatchNorm2d(512)),
            ('relu4',nn.ReLU(True)),

            #5
            ('conv5',nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
            ('relu5',nn.ReLU(True)),
            ('pooling3',nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))),

            #6
            ('conv6',nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0)),
            ('batchnorm6',nn.BatchNorm2d(512)),
            ('relu6',nn.ReLU(True)),
        ]))
        self.cnn = cnn

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)
        
        # add log_softmax to converge output
        output = F.log_softmax(output, dim=2)

        return output


    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0   # replace all nan/inf in gradients to zero

