import os
import shutil
import time

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Variable

from model import ResnetLSTM


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class Classifier(object):

  def __init__(self, config, train_data_loader=None, test_data_loader=None):

    self.print_step = config.print_step

    self.num_epochs = config.num_epochs
    self.input_size = config.input_size
    self.hidden_size = config.hidden_size
    self.dropout = config.dropout
    self.lr = config.lr
    self.num_classes = config.num_classes
    self.best_prec1 = config.best_prec1
    self.resnet_path = config.resnet_path
    self.model_path = config.model_path
    self.pretrained_model = config.pretrained_model
    self.pretrained_model_path = config.pretrained_model_path
    self.train_data_loader = train_data_loader
    self.test_data_loader = test_data_loader
    self.start = 0

    self.build_model()


  def build_model(self):
    # Define a generator and a discriminator
    self.model = ResnetLSTM(self.resnet_path, self.input_size, self.hidden_size, 1, num_classes=self.num_classes, bidirectional=False, dropout=self.dropout)

      # Optimizers
    self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)

    # Print networks
    self.print_network(self.model, 'model')

    if self.pretrained_model:
      checkpoint = torch.load(self.pretrained_model_path)
      self.model.load_state_dict(checkpoint['state_dict'])
      self.start = checkpoint['epoch']

    if torch.cuda.is_available():
      self.model.cuda()


  def print_network(self, model, name):
    num_params = 0
    for p in model.parameters():
      num_params += p.numel()
    print(name)
    print(model)
    print("The number of parameters: {}".format(num_params))


  def train(self):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    self.model.train()

    # The number of iterations per epoch
    iters_per_epoch = len(self.train_data_loader)

    end = time.time()

    for epoch in range(self.start, self.num_epochs):
      for i, (images, words, lengths, labels) in enumerate(self.train_data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = labels
        images = self.to_var(images, volatile=True)
        words = self.to_var(words)
        target_var = self.to_var(target)

        # compute output
        output = self.model(images, words, lengths)

        loss = F.nll_loss(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = self.accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], images.size(0))
        top1.update(prec1[0], images.size(0))
        top5.update(prec5[0], images.size(0))

        # compute gradient and do optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % self.print_step == 0:
          print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            epoch, i, iters_per_epoch, batch_time=batch_time,
            data_time=data_time, loss=losses, top1=top1, top5=top5))

      # evaluate on validation set
      prec1 = self.test()

      # remember best prec@1 and save checkpoint
      is_best = prec1 > self.best_prec1
      best_prec1 = max(prec1, self.best_prec1)
      self.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': self.model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer' : self.optimizer.state_dict(),
      }, is_best)


  def test(self):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    self.model.eval()

    # The number of iterations per epoch
    iters_per_epoch = len(self.test_data_loader)

    end = time.time()
    for i, (images, words, lengths, labels) in enumerate(self.test_data_loader):
      target = labels
      images = self.to_var(images, volatile=True)
      words = self.to_var(words)
      target_var = self.to_var(target)

      # compute output
      output = self.model(images, words, lengths)
      loss = F.nll_loss(output, target_var)

      # measure accuracy and record loss
      prec1, prec5 = self.accuracy(output.data, target, topk=(1, 5))
      losses.update(loss.data[0], images.size(0))
      top1.update(prec1[0], images.size(0))
      top5.update(prec5[0], images.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % self.print_step == 0:
        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
          i, iters_per_epoch, batch_time=batch_time, loss=losses,
          top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


  def save_checkpoint(self, state, is_best):
    print('save_checkpoint')

    checkpoint_file = os.path.join(self.model_path, 'checkpoint.pth')
    best_file = os.path.join(self.model_path, 'best.pth')

    torch.save(state, checkpoint_file)
    if is_best:
      shutil.copyfile(checkpoint_file, best_file)


  def accuracy(self, output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)

    # w*h => h*w
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))

    return res


  def to_var(self, x, volatile=False):
    if torch.cuda.is_available():
      x = x.cuda()
    return Variable(x, volatile=volatile)