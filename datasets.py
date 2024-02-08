import numpy as np
import torch
from torchvision import transforms, datasets


def prepare(args, data_root='./data', train_data_aug=True):

    if args.dataset == 'pets':

        '''
        Setups:
            -Original train and val data are merged, then split into (train, val)
            -Original test data is used as a test split as it is
        '''

        # data augmentation
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # official train+val split
        train_set = datasets.OxfordIIITPet(
            root = data_root, split = 'trainval', 
            transform = transform_train if train_data_aug else transform_test, 
            download = True
        )
        if args.val_heldout > 0:  # re-split if args.val_heldout > 0
            val_set = datasets.OxfordIIITPet(
                root = data_root, split = 'trainval', 
                transform = transform_test, download = True
            )
            val_size = int(args.val_heldout * len(train_set))
            train_size = len(train_set) - val_size
            generator = torch.Generator().manual_seed(args.seed)
            train_set, _ = torch.utils.data.random_split(train_set, [train_size, val_size], generator=generator)
            val_set = torch.utils.data.Subset(val_set, np.setdiff1d(np.arange(len(val_set)), train_set.indices))

        test_set = datasets.OxfordIIITPet(
            root = data_root, split = 'test', 
            transform = transform_test, download = True
        )

        train_loader = torch.utils.data.DataLoader(train_set, 
            batch_size=args.batch_size, shuffle=train_data_aug, pin_memory=args.use_cuda, num_workers=4
        )
        if args.val_heldout > 0:
            val_loader = torch.utils.data.DataLoader(val_set, 
                batch_size=args.batch_size, shuffle=False, pin_memory=args.use_cuda, num_workers=4
            )
        else:
            val_loader = None
        test_loader = torch.utils.data.DataLoader(test_set, 
            batch_size=args.batch_size, shuffle=False, pin_memory=args.use_cuda, num_workers=4
        )

        args.num_classes = 37

    else:

        raise NotImplementedError

    return train_loader, val_loader, test_loader, len(train_set)

