import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aguments.contractive_learning_dataset import ContrastibeLearningDataset
from model.resnet_simclr import ResNetSimCLR
from simclr_model import SimCLR
import argparse

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

def main():
    arg = parser.parse_args()
    n_views = 2
    disable_cuda = True
    data_name = 'cifar10'
    data_dir = './datasets'
    batch_size = 256
    arch = 'resnet18'
    workers = 12
    epochs = 265
    learning_rate = 0.0003
    weight_decay = 1e-4
    out_dim = 128
    log_every_n_steps = 100
    temperature = 0.07

    # Check if gpu training is available
    gpu_index = 0
    if not disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        arg.device = device
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        arg.device = device
        gpu_index = -1

    dataset = ContrastibeLearningDataset(data_dir)

    train_dataset = dataset.get_dataset(data_name, n_views)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = batch_size, shuffle = True, num_workers = workers, pin_memory = True, drop_last = True
    )

    model = ResNetSimCLR(base_model=arch, out_dim=out_dim)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    # It's a no-op if the 'gpu_index' argument is a negative interger or None
    with torch.cuda.device(gpu_index):
        simclr = SimCLR(model = model, optimizer=optimizer, scheduler = scheduler, args = arg)
        simclr.train(train_loader)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()