from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


class Loader(object):
    def __init__(self, dataset_ident, file_path, download, batch_size, data_transform, target_transform, use_cuda):

        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        # This seems to be a bug, but disabling pin memory is way faster when using the GPU
        #kwargs = {}

        # set the dataset
        # NOTE: will need a refractor one we load more different datasets, that require custom classes
        loader_map = {
            'mnist': datasets.MNIST,
            'MNIST': datasets.MNIST,
            'FashionMNIST': datasets.FashionMNIST,
            'fashion': datasets.FashionMNIST
        }

        num_class = {
            'mnist': 10,
            'MNIST': 10,
            'fashion': 10,
            'FashionMNIST': 10
        }

        # Get the datasets
        train_dataset, test_dataset = self.get_dataset(loader_map[dataset_ident], file_path, download,
                                                       data_transform, target_transform)
        # Set the loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

        # infer and set size, idea from:
        # https://github.com/jramapuram/helpers/
        tmp_batch, _ = self.train_loader.__iter__().__next__()
        self.img_shape = list(tmp_batch.size())[1:]
        self.num_class = num_class[dataset_ident]
        # Note, might be incorrect for last batch in an iteration
        self.batch_size = batch_size

    @staticmethod
    def get_dataset(dataset, file_path, download, data_transform, target_transform):

        # Check for transform to be None, a single item, or a list
        # None -> default to transform_list = [transforms.ToTensor()]
        # single item -> list
        if not data_transform:
            data_transform = [transforms.ToTensor()]
        elif not isinstance(data_transform, list):
            data_transform = list(data_transform)

        # Training and Validation datasets
        train_dataset = dataset(file_path, train=True, download=download,
                                transform=transforms.Compose(data_transform),
                                target_transform=target_transform)

        test_dataset = dataset(file_path, train=False, download=download,
                               transform=transforms.Compose(data_transform),
                               target_transform=target_transform)

        return train_dataset, test_dataset


class gauss_circle(object):
    def __init__(self, size=2, num_circles=8, std=0.02, radius=2, uniform=False):
        self.num_circles = num_circles
        self.std = std
        self.size = size
        self.radius = radius
        delta = 2 * np.pi / self.num_circles

        centers_x = np.asarray([self.radius * np.cos(i * delta) for i in range(self.num_circles)])
        centers_y = np.asarray([self.radius * np.sin(i * delta) for i in range(self.num_circles)])

        if self.uniform is True:
            # Strictly Uniform
            self.p = [1. / self.num_circles for _ in range(self.num_circles)]
        else:
            # Random draw from uniform distribution
            self.p = [np.random.uniform() for _ in range(self.num_circles)]
            self.p /= np.sum(self.p)

        self.centers = np.concatenate((centers_x[:, np.newaxis], centers_y[:, np.newaxis]), 1)

    def sample(self, n_samples):
        centers_idx = np.random.choice(self.num_circles, n_samples, p=self.p)
        centers_sample = self.centers[centers_idx, :]
        data_sample = np.random.normal(loc=centers_sample, scale=self.std)
        return data_sample.astype('float32')


class CelebALoader(object):
    """
    loader for the CELEB-A dataset
    """
    def __init__(self, file_path, batch_size, valid_size, shuffle, use_cuda):

        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset, test_dataset = self.get_dataset(file_path, transform)

        # Set the samplers
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # Set the loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, **kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=valid_sampler, **kwargs)

    @staticmethod
    def get_dataset(file_path, transform):

        train_dataset = datasets.ImageFolder(file_path, transform=transform)
        test_dataset = datasets.ImageFolder(file_path, transform=transform)

        return train_dataset, test_dataset

