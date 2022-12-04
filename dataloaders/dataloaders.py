import torch.cuda
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import utils


def load_data(args):
    if "cifar" in args.dataset:
        if args.dataset == "cifar10":
            NUM_CLASS = 10
            mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            std = [x / 255 for x in [63.0, 62.1, 66.7]]
        elif args.dataset == "cifar100":
            NUM_CLASS = 100
            mean = [x / 255 for x in [129.3, 124.1, 112.4]]
            std = [x / 255 for x in [68.2, 65.4, 70.4]]

        SIZE = 32
        NUM_IMAGES = 50000
        NUM_TEST_IMAGES = 10000
        STEPS_PER_EPOCH = int(NUM_IMAGES / args.cnn_batch_size)
        STRIDE = 1
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
    elif args.dataset == "imagenet":
        NUM_CLASS = 1000
        SIZE = 224
        STRIDE = 1
        NUM_IMAGES = int(1281167 // args.cnn_batch_size) * args.cnn_batch_size

        print(
            f"Dropping the last {1281167 - NUM_IMAGES} images in the Train set, since RL environment requires to have"
            f" a fixed batch size."
        )

        NUM_TEST_IMAGES = 100000
        STEPS_PER_EPOCH = int(NUM_IMAGES / args.cnn_batch_size)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        jittering = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        lighting = utils.Lighting(
            alphastd=0.1,
            eigval=[0.2175, 0.0188, 0.0045],
            eigvec=[
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ],
        )
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                jittering,
                lighting,
                normalize,
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

    # Setup dataset
    if args.dataset == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.data_dir + "cifar100",
            train=True,
            download=True,
            transform=train_transform,
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.val_data_dir + "cifar100",
            train=False,
            download=True,
            transform=test_transform,
        )
    elif args.dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir + "cifar10",
            train=True,
            download=True,
            transform=train_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.val_data_dir + "cifar10",
            train=False,
            download=True,
            transform=test_transform,
        )

    elif args.dataset == "imagenet":
        train_dataset = torchvision.datasets.ImageFolder(
            root=args.data_dir, transform=train_transform
        )
        test_dataset = torchvision.datasets.ImageFolder(
            root=args.val_data_dir, transform=test_transform
        )
    else:
        raise NotImplementedError("Unrecognized dataset.")

    d = torch.cuda.get_device_name()
    print(d)
    if "3060" not in d:
        num_workers = torch.cuda.device_count() * 4
    else:
        num_workers = 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.cnn_batch_size,
        drop_last=True,
        shuffle=args.shuffle_data,
        pin_memory=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.cnn_batch_size // torch.cuda.device_count(),
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    return train_loader, test_loader, dict(
        num_class=NUM_CLASS,
        num_images=NUM_IMAGES,
        num_test_images=NUM_TEST_IMAGES,
        steps_per_epoch=STEPS_PER_EPOCH,
        train_transform=train_transform,
        test_transform=test_transform,
        size=SIZE,
        stride=STRIDE
    )
