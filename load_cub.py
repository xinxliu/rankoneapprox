import os
import numpy as np
from PIL import Image
import torch.utils.data as data


def pil_loader(path):
    img = Image.open(path)
    return img.convert("RGB")


def default_loader(path):
    return pil_loader(path)


def build_set(root, train):
    images_file_path = os.path.join(root, 'CUB_200_2011/images/')

    all_images_list_path = os.path.join(root, 'CUB_200_2011/images.txt')
    all_images_list = np.genfromtxt(all_images_list_path, dtype=str)
    train_test_list_path = os.path.join(root, 'CUB_200_2011/train_test_split.txt')
    train_test_list = np.genfromtxt(train_test_list_path, dtype=int)

    imgs = []
    classes = []
    class_to_idx = []

    for i in range(0, len(all_images_list)):
        fname = all_images_list[i, 1]
        full_path = os.path.join(images_file_path, fname)
        if train_test_list[i, 1] == 1 and train:
            imgs.append((full_path, int(fname[0:3]) - 1))
        elif train_test_list[i, 1] == 0 and not train:
            imgs.append((full_path, int(fname[0:3]) - 1))
        if os.path.split(fname)[0][4:] not in classes:
            classes.append(os.path.split(fname)[0][4:])
            class_to_idx.append(int(fname[0:3]) - 1)

    return imgs, classes, class_to_idx


class CUB200(data.Dataset):

    def __init__(self, root='/opt/liuxx/Dataset/CUB_200_2011', train=True, transform=None, target_transform=None,
                 loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.loader = loader
        self.urls = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'

        if not self._check_exists():
            raise RuntimeError('No Dataset! u can download it from' + self.urls)

        self.imgs, self.classes, self.class_to_idx = build_set(self.root, self.train)

    def __getitem__(self, index):

        path, target = self.imgs[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_exists(self):
        pth = self.root
        return os.path.exists(os.path.join(pth, 'CUB_200_2011/'))

    def __len__(self):
        return len(self.imgs)
