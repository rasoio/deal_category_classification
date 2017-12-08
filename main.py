import os
import argparse
from torch.backends import cudnn
from data_loader import get_loader
from classifier import Classifier

def main(config):
  # For fast training
  cudnn.benchmark = True

  # Create directories if not exist
  if not os.path.exists(config.log_path):
    os.makedirs(config.log_path)
  if not os.path.exists(config.model_path):
    os.makedirs(config.model_path)

  # Mode
  if config.mode == 'train':

    train_data_loader = get_loader(config.train_csv_path, config.train_img_path, config.category_file_path,
                              config.batch_size, config.word2vec_path, config.input_size, config.num_workers, config.mode)

    test_data_loader = get_loader(config.valid_csv_path, config.valid_img_path, config.category_file_path,
                             config.batch_size, config.word2vec_path, config.input_size, config.num_workers, config.mode)

    classifier = Classifier(config, train_data_loader=train_data_loader, test_data_loader=test_data_loader)

    classifier.train()
  elif config.mode == 'test':

    test_data_loader = get_loader(config.valid_csv_path, config.valid_img_path, config.category_file_path,
                             config.batch_size, config.word2vec_path, config.input_size, config.num_workers, config.mode)

    classifier = Classifier(config, test_data_loader=test_data_loader)

    classifier.test()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # Model hyper-parameters
  parser.add_argument('--input_size', type=int, default=100)
  parser.add_argument('--hidden_size', type=int, default=512)
  parser.add_argument('--num_classes', type=int, default=1000)
  parser.add_argument('--momentum', type=float, default=0.01)

  # Training settings
  parser.add_argument('--num_epochs', type=int, default=20)
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--num_workers', type=int, default=1)

  parser.add_argument('--best_prec1', type=int, default=0)
  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--dropout', type=float, default=0.75)

  # Misc
  parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

  # Path
  parser.add_argument('--word2vec_path', type=str, default='./data/word2vec/model.dat')
  parser.add_argument('--train_csv_path', type=str, default='./data/train.csv')
  parser.add_argument('--train_img_path', type=str, default='./data/imgs/train')
  parser.add_argument('--valid_csv_path', type=str, default='./data/valid.csv')
  parser.add_argument('--valid_img_path', type=str, default='./data/imgs/valid')
  parser.add_argument('--category_file_path', type=str, default='./data/categories.txt')
  parser.add_argument('--resnet_path', type=str, default='./models/model_best_resnet.pth')
  parser.add_argument('--log_path', type=str, default='./logs')
  parser.add_argument('--pretrained_model', type=bool, default=False)
  parser.add_argument('--pretrained_model_path', type=str, default='./models/')
  parser.add_argument('--model_path', type=str, default='./models')

  # Step size
  parser.add_argument('--print_step', type=int, default=10)

  config = parser.parse_args()
  print(config)
  main(config)

