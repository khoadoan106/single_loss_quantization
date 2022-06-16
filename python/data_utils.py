import torch.utils.data as util_data
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class MyCIFAR10(dsets.CIFAR10):
    def __init__(self,
            root,
            train,
            transform=None,
            target_transform=None,
            download=False,
            n_class=10):
        super().__init__(root, train, transform, target_transform, download)
        self.n_class = n_class
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(self.n_class, dtype=np.int8)[np.array(target)]
        return img, target, index
    
def create_cifar_unseen_dataset(args, train_n_class=7):
    """Create unseen retrieval datasets
       train_n_class: number of classes used for training
    """
    batch_size = args.batch_size
    
    train_selected_labels = list(range(train_n_class))
    test_selected_labels = list(range(train_n_class, 10))


    train_size = 500
    test_size = 100

    transform = transforms.Compose([
        transforms.Resize(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Dataset
    train_dataset = MyCIFAR10(root=args.data_path, train=True, transform=transform, download=True)
    test_dataset = MyCIFAR10(root=cifar_dataset_root, train=False, transform=transform)
    database_dataset = MyCIFAR10(root=cifar_dataset_root, train=False, transform=transform)

    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    np.random.seed(args.random_seed)
    first = True
    for label in train_selected_labels:
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            train_index = index[test_size: train_size + test_size]
        else:
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
        first = False
        
    first = True
    for label in test_selected_labels:
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False   

    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    train_dataset.n_class = train_n_class
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]-train_n_class
    test_dataset.n_class = 10-train_n_class
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]-train_n_class
    database_dataset.n_class = 10-train_n_class

    print("Train dataset: {} samples, {} classes".format(train_dataset.data.shape[0], train_dataset.n_class))
    print("Test dataset: {} samples, {} classes".format(test_dataset.data.shape[0], test_dataset.n_class))
    print("Dataset dataset: {} samples, {} classes".format(database_dataset.data.shape[0], database_dataset.n_class))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=args.workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.workers)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=args.workers)

    return train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]

def cifar_dataset(args):
    batch_size = args.batch_size

    train_size = 500
    test_size = 100

    if args.dataset == "cifar10-2":
        train_size = 5000
        test_size = 1000

    transform = transforms.Compose([
        transforms.Resize(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Dataset
    train_dataset = MyCIFAR10(root=args.data_path, train=True, transform=transform, download=True)
    test_dataset = MyCIFAR10(root=args.data_path, train=False, transform=transform)
    database_dataset = MyCIFAR10(root=args.data_path, train=False, transform=transform)

    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))
    
    np.random.seed(args.random_seed)
    first = True
    for label in range(10):
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    if args.dataset == "cifar10":
        # test:1000, train:5000, database:54000
        pass
    elif args.dataset == "cifar10-1":
        # test:1000, train:5000, database:59000
        database_index = np.concatenate((train_index, database_index))
    elif args.dataset == "cifar10-2":
        # test:10000, train:50000, database:50000
        database_index = train_index

    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=args.workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              pin_memory=True,
                                              shuffle=False,
                                              num_workers=args.workers)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  pin_memory=True,
                                                  shuffle=False,
                                                  num_workers=args.workers)
    
    
    print("Train dataset: {} samples".format(train_dataset.data.shape[0]))
    print("Test dataset: {} samples".format(test_dataset.data.shape[0]))
    print("Dataset dataset: {} samples".format(database_dataset.data.shape[0]))
    

    return train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]


def cifar_sub_dataset(args, n_class=2):
    batch_size = args.batch_size
    selected_labels = list(range(n_class))

    train_size = 500
    test_size = 100

    transform = transforms.Compose([
        transforms.Resize(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset
    train_dataset = MyCIFAR10(root=args.data_path, train=True, transform=transform, download=True, n_class=n_class)
    test_dataset = MyCIFAR10(root=args.data_path, train=False,transform=transform, n_class=n_class)
    database_dataset = MyCIFAR10(root=args.data_path, train=False, transform=transform, n_class=n_class)

    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    first = True
    np.random.seed(args.random_seed)
    for label in selected_labels:
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False    

    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    print("Train dataset: {} samples".format(train_dataset.data.shape[0]))
    print("Test dataset: {} samples".format(test_dataset.data.shape[0]))
    print("Dataset dataset: {} samples".format(database_dataset.data.shape[0]))
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=args.workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.workers)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=args.workers)

    return train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]

class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)
    
    
def image_transform(resize_size, crop_size, data_set):
    if data_set == "data_train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])    
    
def get_data(args):
    if 'ucifar10' == args.dataset:
        return cifar_unseen_dataset(args, train_n_class=7)
    elif "cifar10" in args.dataset:
        return cifar_dataset(args)
    elif args.dataset == 'cifar2':
        return cifar_sub_dataset(args, n_class=2)
    elif args.dataset == 'cifar4':
        return cifar_sub_dataset(args, n_class=4)

    dsets = {}
    dset_loaders = {}

    for data_set in ["data_train_set", "data_test_set", "data_database_set"]:
        dsets[data_set] = ImageList(args.data_path,
                                    open(args.__dict__[data_set]).readlines(),
                                    transform=image_transform(args.resize_size, args.crop_size, data_set))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=args.batch_size,
                                                      pin_memory=True,
                                                      shuffle=True, num_workers=args.workers)
        if dsets[data_set].transform is not None:
            print(dsets[data_set].transform)

    return dset_loaders["data_train_set"], dset_loaders["data_test_set"], dset_loaders["data_database_set"], \
           len(dsets["data_train_set"]), len(dsets["data_test_set"]), len(dsets["data_database_set"])        
