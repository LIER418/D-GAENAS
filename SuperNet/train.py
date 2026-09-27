from supernet_train import Trainer
import argparse

parser = argparse.ArgumentParser(description='OneShot SuperNet')
parser.add_argument('--dataset', type=str, default='CIFAR10',
                    help='DataSet')
parser.add_argument('--batch_size', type=int, default=50, 
                        help='Batch Size')
parser.add_argument('--epochs', type=int, default=500, 
                        help='Number of Training Epochs of Model')
parser.add_argument('--cuda', type=int, default=0, 
                        help='CUDA Device Number')
parser.add_argument('--data_dir', type=str, default='../data',
                    help='CIFAR data directory')
parser.add_argument('--no_download', action='store_true',
                    help='Do not download CIFAR data automatically')
args = parser.parse_args()

trainer = Trainer(dataset=args.dataset, batch_size=args.batch_size,
                  init_channels=48, epochs=args.epochs, gpu=args.cuda,
                  data_root=args.data_dir, download=not args.no_download)
trainer.run()
