import os
from io import open

import numpy as np
import pandas as pd
import torch
from PIL import Image
from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DealDataset(Dataset):
  """Deal dataset."""

  def __init__(self, csv_file, category_file_path, root_dir, word2vec_path, input_size, transform=None):
    """
    Args:
        csv_file (string): Path to the csv file with annotations.
        cid_file (string): Path to the cids file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """

    self.deals_frame = pd.read_csv(csv_file, sep='\t',
                                   encoding='utf8',
                                   quoting=3, # QUOTE_NONE
                                   error_bad_lines=False,
                                   names=['DID','CID','WORDS'])

    #testing
    #         self.deals_frame = self.deals_frame.sample(frac=0.1)
    #         self.index_values = self.deals_frame.index.values

    categories, cate_to_idx = read_categories(category_file_path)
    self.categories = categories
    self.cate_to_idx = cate_to_idx
    self.root_dir = root_dir
    self.transform = transform

    word2vec_model = Word2Vec.load(word2vec_path)
    self.word_vectors = word2vec_model.wv
    del word2vec_model

    self.input_size = input_size

  def __len__(self):
    return len(self.deals_frame)

  def __getitem__(self, idx):

    #         idx = self.index_values[idx]

    did = str(self.deals_frame.loc[idx, 'DID'])
    cid = self.deals_frame.loc[idx, 'CID']
    words = self.deals_frame.loc[idx, 'WORDS'] # words tokens + word2vec

    file_name = did + '.jpg'

    img_path = os.path.join(self.root_dir, cid, file_name)

    # before remove error images
    image = Image.open(img_path).convert('RGB')

    if self.transform is not None:
      image = self.transform(image)

    words_tensor = self.lineToTensor(words)

    label = self.cate_to_idx[cid]

    return (image, words_tensor, label)

  # Turn a line into a <line_length x word2vec(100)>,
  def lineToTensor(self, line):

    words = line.split()

    tensor = torch.zeros(len(words), self.input_size)
    for li, word in enumerate(words):
      try:
        tensor[li] = torch.from_numpy(np.array(self.word_vectors.wv[word]))
      except:
        tensor[li] = torch.ones(self.input_size)

    return tensor


def get_loader(csv_path, img_path, category_file_path, batch_size, word2vec_path, input_size, num_workers=4, mode='train'):
  """Build and return data loader."""

  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

  if mode == 'train':
    transform = transforms.Compose([
      transforms.RandomSizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize
    ])
  else:
    transform = transforms.Compose([
      transforms.Scale(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ])

  dataset = DealDataset(csv_file=csv_path,
                        category_file_path=category_file_path,
                        root_dir=img_path,
                        transform=transform,
                        word2vec_path=word2vec_path,
                        input_size=input_size)

  shuffle = False
  if mode == 'train':
    shuffle = True

  data_loader = DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           shuffle=shuffle,
                           collate_fn=collate_fn)
  return data_loader


def collate_fn(data):
  """Creates mini-batch tensors from the list of tuples (image, caption).

  We should build custom collate_fn rather than using default collate_fn,
  because merging caption (including padding) is not supported in default.
  Args:
      data: list of tuple (image, caption).
          - image: torch tensor of shape (3, 256, 256).
          - caption: torch tensor of shape (?); variable length.
  Returns:
      images: torch tensor of shape (batch_size, 3, 256, 256).
      targets: torch tensor of shape (batch_size, padded_length).
      lengths: list; valid length for each padded caption.
      labels: list; cid.
  """
  # Sort a data list by caption length (descending order).
  data.sort(key=lambda x: len(x[1]), reverse=True)
  #     print(data)
  images, words, labels = zip(*data)

  labels = torch.Tensor([label for label in labels]).long()

  # Merge images (from tuple of 3D tensor to 4D tensor).
  images = torch.stack(images, 0)

  # Merge captions (from tuple of 1D tensor to 2D tensor).
  lengths = [len(word) for word in words]

  #     print('images', images.size())
  targets = torch.zeros(len(words), max(lengths),100)

  #     print('targets', targets.size())
  for i, word in enumerate(words):
    end = lengths[i]
    targets[i, :end] = word[:end]
  return images, targets, lengths, labels



def read_categories(category_file_path):
  categories = open(category_file_path, encoding='utf-8').read().strip().split('\n')
  cate_to_idx = {categories[i]: i for i in range(len(categories))}
  return categories, cate_to_idx