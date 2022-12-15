from torchvision.transforms import transforms
from data_aguments.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aguments.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDataetSelection

class ContrastibeLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s = 1):
        """Return a set of data agumentation transformations as described in the SimCLR paper"""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_tramsforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p = 0.8),
                                              transforms.RandomGrayscale(p = 0.2),
                                              # transforms.GaussianBlur(kernel_size=int(0,1 * size)),
                                              GaussianBlur(kernel_size=int(0.1*size)),
                                              transforms.ToTensor()])

        return data_tramsforms

    def get_dataset(self, name, n_views):
        valid_dataset = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                             transform=ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(32), n_views),
                                                             download=True),
                         'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled', transform=ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(96), n_views),
                                                         download=True)}
        try:
            dataset_fn = valid_dataset[name]

        except KeyError:
            raise InvalidDataetSelection()
        else:
            return dataset_fn()