import random
import numpy as np
from PIL import Image, ImageFile

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from utility import RandomIdentitySampler, RandomErasing3
from Datasets.MARS_dataset import Mars
from Datasets.iLDSVID import iLIDSVID
from Datasets.PRID_dataset import PRID

ImageFile.LOAD_TRUNCATED_IMAGES = True

__factory = {
    'Mars': Mars,
    'iLIDSVID': iLIDSVID,
    'PRID': PRID
}


def train_collate_fn(batch):
    imgs, pids, camids, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    labels = torch.stack(labels, dim=0)
    return imgs, pids, camids, labels


def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    imgs = torch.stack(imgs, dim=0)              # [1, num_clips, seq_len, C, H, W] when batch_size=1
    camids = torch.tensor(camids, dtype=torch.int64)
    return imgs, pids, camids, img_paths


def dataloader(Dataset_name):
    train_transforms = T.Compose([
        T.Resize([256, 128], interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = __factory[Dataset_name]()

    train_set = VideoDataset_inderase(
        dataset.train,
        seq_len=4,
        sample='intelligent',
        transform=train_transforms
    )

    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    train_loader = DataLoader(
        train_set,
        batch_size=64,
        sampler=RandomIdentitySampler(dataset.train, 64, 4),
        num_workers=4,
        collate_fn=train_collate_fn,
        pin_memory=True
    )

    q_val_set = VideoDataset(
        dataset.query,
        seq_len=4,
        sample='dense',
        transform=val_transforms
    )
    g_val_set = VideoDataset(
        dataset.gallery,
        seq_len=4,
        sample='dense',
        transform=val_transforms
    )

    # IMPORTANT:
    # return validation DATALOADERS, not raw datasets
    q_val_loader = DataLoader(
        q_val_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=val_collate_fn,
        pin_memory=True
    )

    g_val_loader = DataLoader(
        g_val_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=val_collate_fn,
        pin_memory=True
    )

    return train_loader, len(dataset.query), num_classes, cam_num, view_num, q_val_loader, g_val_loader


def read_image(img_path):
    """Keep reading image until succeed."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def _pad_indices(indices, seq_len):
    indices = list(indices)
    if len(indices) == 0:
        raise RuntimeError("Empty tracklet encountered.")
    while len(indices) < seq_len:
        indices.append(indices[-1])
    return indices


class VideoDataset(Dataset):
    """
    Video Person ReID Dataset.
    Training sample: (seq_len, C, H, W)
    Dense test sample: (num_clips, seq_len, C, H, W)
    """

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None, max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        if self.sample == 'random':
            frame_indices = list(range(num))
            rand_end = max(0, len(frame_indices) - self.seq_len)
            begin_index = random.randint(0, rand_end)
            indices = frame_indices[begin_index:begin_index + self.seq_len]
            indices = _pad_indices(indices, self.seq_len)

            imgs = []
            targt_cam = []
            for idx in indices:
                img_path = img_paths[int(idx)]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                imgs.append(img.unsqueeze(0))
                targt_cam.append(camid)

            imgs = torch.cat(imgs, dim=0)
            return imgs, pid, targt_cam

        elif self.sample == 'dense':
            """
            Split full tracklet into a list of clips, each clip has seq_len frames.
            Used for evaluation. DataLoader batch_size must be 1.
            """
            frame_indices = list(range(num))
            indices_list = []

            cur_index = 0
            while cur_index + self.seq_len <= num:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len

            if cur_index < num:
                last_seq = frame_indices[cur_index:]
                last_seq = _pad_indices(last_seq, self.seq_len)
                indices_list.append(last_seq)

            if len(indices_list) == 0:
                indices_list.append(_pad_indices(frame_indices, self.seq_len))

            imgs_list = []
            for indices in indices_list[:self.max_length]:
                imgs = []
                for idx in indices:
                    img_path = img_paths[int(idx)]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    imgs.append(img.unsqueeze(0))
                imgs = torch.cat(imgs, dim=0)   # [seq_len, C, H, W]
                imgs_list.append(imgs)

            imgs_array = torch.stack(imgs_list, dim=0)  # [num_clips, seq_len, C, H, W]
            return imgs_array, pid, camid, img_paths

        elif self.sample == 'dense_subset':
            frame_indices = list(range(num))
            rand_end = max(0, len(frame_indices) - self.max_length)
            begin_index = random.randint(0, rand_end) if rand_end > 0 else 0

            indices_list = []
            cur_index = begin_index
            while cur_index + self.seq_len <= num:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len

            if cur_index < num:
                last_seq = frame_indices[cur_index:]
                last_seq = _pad_indices(last_seq, self.seq_len)
                indices_list.append(last_seq)

            if len(indices_list) == 0:
                indices_list.append(_pad_indices(frame_indices, self.seq_len))

            imgs_list = []
            for indices in indices_list[:self.max_length]:
                imgs = []
                for idx in indices:
                    img_path = img_paths[int(idx)]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    imgs.append(img.unsqueeze(0))
                imgs = torch.cat(imgs, dim=0)
                imgs_list.append(imgs)

            imgs_array = torch.stack(imgs_list, dim=0)
            return imgs_array, pid, camid

        elif self.sample == 'intelligent_random':
            indices = []
            each = max(num // self.seq_len, 1)
            for i in range(self.seq_len):
                if i != self.seq_len - 1:
                    left = min(i * each, num - 1)
                    right = min((i + 1) * each - 1, num - 1)
                else:
                    left = min(i * each, num - 1)
                    right = num - 1
                indices.append(random.randint(left, right))

            imgs = []
            for idx in indices:
                img_path = img_paths[int(idx)]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                imgs.append(img.unsqueeze(0))
            imgs = torch.cat(imgs, dim=0)
            return imgs, pid, camid

        else:
            raise KeyError("Unknown sample method: {}".format(self.sample))


class VideoDataset_inderase(Dataset):
    """
    Video Person ReID Dataset with random erasing labels for training.
    """

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None, max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length
        self.erase = RandomErasing3(probability=0.5, mean=[0.485, 0.456, 0.406])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        if self.sample != "intelligent":
            frame_indices = list(range(num))
            rand_end = max(0, len(frame_indices) - self.seq_len)
            begin_index = random.randint(0, rand_end)
            indices = frame_indices[begin_index:begin_index + self.seq_len]
            indices = _pad_indices(indices, self.seq_len)
        else:
            indices = []
            each = max(num // self.seq_len, 1)
            for i in range(self.seq_len):
                if i != self.seq_len - 1:
                    left = min(i * each, num - 1)
                    right = min((i + 1) * each - 1, num - 1)
                else:
                    left = min(i * each, num - 1)
                    right = num - 1
                indices.append(random.randint(left, right))

        imgs = []
        labels = []
        targt_cam = []

        for idx in indices:
            img_path = img_paths[int(idx)]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img, temp = self.erase(img)
            labels.append(temp)
            imgs.append(img.unsqueeze(0))
            targt_cam.append(camid)

        labels = torch.tensor(labels)
        imgs = torch.cat(imgs, dim=0)

        return imgs, pid, targt_cam, labels
