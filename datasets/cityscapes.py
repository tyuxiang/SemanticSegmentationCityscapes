import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split

import os
from PIL import Image
import numpy as np

from datasets.labels import label2trainid
from preprocessing.augment_script import get_augmented_filename

LABEL_POSTFIX = '_gtFine_labelIds.png'

def setupDatasetsAndLoaders(dir, batch_size=64, use_augmented=False):
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ''' init transforms here '''
    train_transform = transforms.Compose([
        transforms.Resize((224,448)),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224,448)),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224,448)),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std),
    ])
    inputTransforms = [train_transform, val_transform, test_transform]

    train_set, val_set, test_set = makeDatasets(dir, inputTransforms=inputTransforms, use_augmented=use_augmented)

    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader

def makeDatasets(dataset_dir, inputTransforms, rnd_seed=42, use_augmented=False):
    items = findAllItems(dataset_dir)
    # trainvaltest split
    n = len(items)
    ratios = [int(0.7*n), int(0.15*n), n - int(0.7*n) - int(0.15*n)]
    torch.manual_seed(rnd_seed)
    splitIdx = random_split(range(n), ratios)

    modes = ['train', 'val', 'test']
    dss = []
    for modeIdx, mode in enumerate(modes):
        modeItems = [items[index] for index in splitIdx[modeIdx]]
        ds = CityScapes(mode, modeItems, inputTransforms[modeIdx], use_augmented=use_augmented)
        dss.append(ds)

    train_set, val_set, test_set = dss
    return train_set, val_set, test_set

def makeSequencePaths(dir, img_name, seq_dir, city, k=4):
    name_split = img_name.split('_')
    seq = '_'.join(name_split[:2])
    frame = int(name_split[2])

    frames = []
    for i in range(frame-k+1, frame+1):
        frames.append(os.path.join(dir, seq_dir, city, '{}_{:06d}_leftImg8bit.png'.format(seq, i)))

    return frames

def findAllItems(dir):
    in_dir = 'leftImg8bit'
    out_dir = 'gtFine'
    seq_dir = 'leftImg8bit_sequence'

    items = []
    cities = [c for c in os.listdir(os.path.join(dir, in_dir))]
    # just to make sure the items indices are deterministic
    cities.sort()

    for city in cities:

        img_paths = os.listdir(os.path.join(dir, in_dir, city))
        # just to make sure the items indices are deterministic
        img_paths.sort()

        for img_path in img_paths:
            seq = int(img_path.split('_')[1])
            if seq >= 900000:
                continue
            
            in_path = os.path.join(dir, in_dir, city, img_path)

            img_name = img_path.split('_leftImg8bit.png')[0]
            out_path = os.path.join(dir, out_dir, city, img_name+LABEL_POSTFIX)

            seq_paths = makeSequencePaths(dir, img_name, seq_dir, city)

            items.append((seq_paths, out_path, in_path))
    return items

def addAugmentedItems(items: list):
    res = items.copy()

    for idx, item in enumerate(items):
        in_path = item[-1]
        aug_path = get_augmented_filename(in_path)
        
        seq_paths = item[0]
        aug_seq = []
        for s in seq_paths:
            aug_seq.append(get_augmented_filename(s))

        resItem = (aug_seq, item[1], aug_path)
        res.append(resItem)

    return res

class CityScapes(Dataset):
    def __init__(self, mode, items, transform, use_augmented=False):
        self.mode = mode
        self.transform = transform

        if self.mode == 'train' and use_augmented:
            self.items = addAugmentedItems(items)
        else:
            self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        seq_paths, out_path, in_path = self.items[index]
        img_name = os.path.splitext(os.path.basename(in_path))[0]

        images = []
        for path in seq_paths:
            image = Image.open(path).convert('RGB')

            if self.transform:
                image = self.transform(image)
            
            images.append(image)

        output = Image.open(out_path)
        t = transforms.Resize((224,448))
        output = t(output)
        output = np.array(output)

        temp = output.copy()
        # converting labelIDs to trainIDs
        for k,v in label2trainid.items():
            temp[output == k] = v
        
        output = torch.from_numpy(temp)

        return images, output, img_name
