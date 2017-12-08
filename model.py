import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence


class ResnetLSTM(nn.Module):

  # embed_size : 100, hidden_size : 512, num_layers : 1
  def __init__(self, resnet_path, embed_size, hidden_size, num_layers=1, num_classes=10, momentum=0.01, dropout=0, bidirectional=False):
    super(ResnetLSTM, self).__init__()

    self.num_classes = num_classes

    # pretrained load
    #         resnet = models.resnet152(pretrained=True)
    resnet = self.load_resnet(resnet_path, num_classes) # resnet trained num_classes

    modules = list(resnet.children())[:-1]      # delete the last fc layer.

    self.resnet = nn.Sequential(*modules)

    #         print(resnet)
    # resnet.fc.in_features = 512
    self.linear = nn.Linear(resnet.fc.in_features, embed_size)
    self.bn = nn.BatchNorm1d(embed_size, momentum=momentum)
    #         self.init_weights()

    #lstm
    self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                        batch_first=True, dropout=dropout, bidirectional=bidirectional)

    self.hidden2cid = nn.Linear(hidden_size, num_classes)

    self.softmax = nn.LogSoftmax()


  def load_resnet(self, resnet_path, num_classes):

    if torch.cuda.is_available():
      resnet = models.resnet18(pretrained=True)
      resnet = torch.nn.DataParallel(resnet).cuda()

      # resnet = models.resnet18(pretrained=False, num_classes=num_classes)
      # checkpoint = torch.load(resnet_path)
      # resnet.load_state_dict(checkpoint['state_dict'])
      return resnet.module
    else:
      resnet = models.resnet18(pretrained=True)
      return resnet


  #     def init_weights(self):
  #         """Initialize the weights."""
  #         self.linear.weight.data.normal_(0.0, 0.02)
  #         self.linear.bias.data.fill_(0)


  def forward(self, images, words, lengths):
    """Extract the image feature vectors."""

    #         print(images.size())
    features = self.resnet(images)

    features = features.view(features.size(0), -1)

    features = self.bn(self.linear(features))

    # clone features
    images = Variable(features.unsqueeze(1).data)

    if torch.cuda.is_available():
      images.cuda()

    #         print('features', features.size())
    # print('images', images.size())
    # print('words', words.size())
    embeddings = torch.cat((images, words), 1)

    #         print('embeddings', embeddings.size())

    # pad seq
    # ('embeddings', torch.Size([2, 17, 100]))
    # batch_size * sentences(n * 100)
    packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

    # PackedSequence
    output, (h, c) = self.lstm(packed)

    # last hidden state !!
    # https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e
    last = h[-1]

    cid_space = self.hidden2cid(last)
    outputs = self.softmax(cid_space)
    #         print('outputs', outputs.size(), outputs.requires_grad)
    return outputs
