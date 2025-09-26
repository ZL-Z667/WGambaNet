from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image
import nibabel as nib

class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self)
        if train:
            images_list = sorted(os.listdir(path_Data+'train/images/'))
            masks_list = sorted(os.listdir(path_Data+'train/masks/'))
            # images_list = sorted(os.listdir(path_Data + 'train/images/'), key=lambda x: int(os.path.splitext(x)[0]))
            # masks_list = sorted(os.listdir(path_Data + 'train/masks/'), key=lambda x: int(os.path.splitext(x)[0]))

            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'train/images/' + images_list[i]
                mask_path = path_Data+'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer
        else:
            # images_list = sorted(os.listdir(path_Data + 'val/images/'), key=lambda x: int(os.path.splitext(x)[0]))
            # masks_list = sorted(os.listdir(path_Data + 'val/masks/'), key=lambda x: int(os.path.splitext(x)[0]))
            images_list = sorted(os.listdir(path_Data + 'val/images/'))
            masks_list = sorted(os.listdir(path_Data + 'val/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'val/images/' + images_list[i]
                mask_path = path_Data+'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer

    # def __getitem__(self, indx):
    #     img_path, msk_path = self.data[indx]
    #     img = np.array(Image.open(img_path).convert('RGB'))
    #     msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
    #     img, msk = self.transformer((img, msk))
    #     return img, msk

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        img_name = os.path.basename(img_path)
        return img, msk, img_name

    def __len__(self):
        return len(self.data)
    


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size, input_channels=1):
        self.output_size = output_size
        self.input_channels = input_channels

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if isinstance(label, torch.Tensor):
            label = label.numpy()

        if image.ndim == 2:  # (H, W)
            if self.input_channels == 3:
                image = np.stack([image]*3, axis=0)  # (3, H, W)
            else:
                image = image[np.newaxis, ...]       # (1, H, W)
        elif image.ndim == 3:
            if image.shape[0] == 1 and self.input_channels == 3:
                image = np.repeat(image, 3, axis=0) # (3, H, W)

        if label.ndim == 3:
            label = label[0]

        x, y = image.shape[-2], image.shape[-1]

        if random.random() > 0.5:
            for i in range(image.shape[0]):
                image[i], label = random_rot_flip(image[i], label)
        elif random.random() > 0.5:
            for i in range(image.shape[0]):
                image[i], label = random_rotate(image[i], label)

        if x != self.output_size[0] or y != self.output_size[1]:
            for i in range(image.shape[0]):
                image[i] = zoom(image[i], (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

'''
# for acdc
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample
'''

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

"""
class ACDC_dataset(Dataset):
    def __init__(self, base_dir, split, transform=None):
        self.transform = transform
        self.split = split
        split_mapping = {
            'train': 'train',
            'val': 'val',
            'test': 'test'
        }
        self.data_dir = os.path.join(base_dir, split_mapping[split])
        self.patient_folders = sorted([f for f in os.listdir(self.data_dir) if f.startswith('patient')])
        self.samples = []
        for patient in self.patient_folders:
            patient_path = os.path.join(self.data_dir, patient)
            files = os.listdir(patient_path)
            gt_files = [f for f in files if '_gt' in f]
            for gt_file in gt_files:
                image_file = gt_file.replace('_gt', '')
                # 读取体积，遍历所有切片
                image = nib.load(os.path.join(patient_path, image_file)).get_fdata()
                label = nib.load(os.path.join(patient_path, gt_file)).get_fdata()
                num_slices = image.shape[2]
                for slice_idx in range(num_slices):
                    self.samples.append({
                        'image_path': os.path.join(patient_path, image_file),
                        'label_path': os.path.join(patient_path, gt_file),
                        'patient_id': patient,
                        'frame_num': gt_file.split('_')[1].replace('frame', ''),
                        'slice_idx': slice_idx
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = nib.load(sample['image_path']).get_fdata()
        label = nib.load(sample['label_path']).get_fdata()
        slice_idx = sample['slice_idx']
        image_slice = image[:, :, slice_idx]
        label_slice = label[:, :, slice_idx]
        target_size = (224, 224)
        if image_slice.shape != target_size:
            image_slice = zoom(image_slice, (target_size[0] / image_slice.shape[0], target_size[1] / image_slice.shape[1]), order=3)
            label_slice = zoom(label_slice, (target_size[0] / label_slice.shape[0], target_size[1] / label_slice.shape[1]), order=0)
        image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
        sample_dict = {
            'image': torch.from_numpy(image_slice).float().unsqueeze(0),
            'label': torch.from_numpy(label_slice).long(),
            'patient_id': sample['patient_id'],
            'frame_num': sample['frame_num'],
            'slice_idx': slice_idx
        }
        if self.transform:
            sample_dict = self.transform(sample_dict)
        return sample_dict
"""
class ACDC_dataset(Dataset):
    def __init__(self, base_dir, split, transform=None):
        self.transform = transform
        self.split = split
        split_mapping = {
            'train': 'train',
            'val': 'val',
            'test': 'test'
        }
        self.data_dir = os.path.join(base_dir, split_mapping[split])
        self.patient_folders = sorted([f for f in os.listdir(self.data_dir) if f.startswith('patient')])

        self.samples = []
        if split == 'train':

            for patient in self.patient_folders:
                patient_path = os.path.join(self.data_dir, patient)
                files = os.listdir(patient_path)
                gt_files = [f for f in files if '_gt' in f]
                for gt_file in gt_files:
                    image_file = gt_file.replace('_gt', '')
                    image = nib.load(os.path.join(patient_path, image_file)).get_fdata()
                    label = nib.load(os.path.join(patient_path, gt_file)).get_fdata()
                    num_slices = image.shape[2]
                    for slice_idx in range(num_slices):
                        self.samples.append({
                            'image_path': os.path.join(patient_path, image_file),
                            'label_path': os.path.join(patient_path, gt_file),
                            'patient_id': patient,
                            'frame_num': gt_file.split('_')[1].replace('frame', ''),
                            'slice_idx': slice_idx
                        })
        else:

            for patient in self.patient_folders:
                patient_path = os.path.join(self.data_dir, patient)
                files = os.listdir(patient_path)
                gt_files = [f for f in files if '_gt' in f]
                for gt_file in gt_files:
                    image_file = gt_file.replace('_gt', '')
                    self.samples.append({
                        'image_path': os.path.join(patient_path, image_file),
                        'label_path': os.path.join(patient_path, gt_file),
                        'patient_id': patient,
                        'frame_num': gt_file.split('_')[1].replace('frame', '')
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.split == 'train':
            image = nib.load(sample['image_path']).get_fdata()
            label = nib.load(sample['label_path']).get_fdata()
            slice_idx = sample['slice_idx']
            image_slice = image[:, :, slice_idx]
            label_slice = label[:, :, slice_idx]
            target_size = (224, 224)
            if image_slice.shape != target_size:
                image_slice = zoom(image_slice, (target_size[0] / image_slice.shape[0], target_size[1] / image_slice.shape[1]), order=3)
                label_slice = zoom(label_slice, (target_size[0] / label_slice.shape[0], target_size[1] / label_slice.shape[1]), order=0)
            image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
            sample_dict = {
                'image': torch.from_numpy(image_slice).float().unsqueeze(0),
                'label': torch.from_numpy(label_slice).long(),
                'patient_id': sample['patient_id'],
                'frame_num': sample['frame_num'],
                'slice_idx': slice_idx
            }
            if self.transform:
                sample_dict = self.transform(sample_dict)
            return sample_dict
        else:

            image = nib.load(sample['image_path']).get_fdata()
            label = nib.load(sample['label_path']).get_fdata()
            target_size = (224, 224)
            num_slices = image.shape[2]

            images = []
            labels = []
            for i in range(num_slices):
                img_slice = image[:, :, i]
                lbl_slice = label[:, :, i]
                if img_slice.shape != target_size:
                    img_slice = zoom(img_slice, (target_size[0] / img_slice.shape[0], target_size[1] / img_slice.shape[1]), order=3)
                    lbl_slice = zoom(lbl_slice, (target_size[0] / lbl_slice.shape[0], target_size[1] / lbl_slice.shape[1]), order=0)
                img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
                images.append(img_slice)
                labels.append(lbl_slice)
            images = torch.from_numpy(np.stack(images)).float().unsqueeze(1) # [num_slices, 1, H, W]
            labels = torch.from_numpy(np.stack(labels)).long() # [num_slices, H, W]
            sample_dict = {
                'image': images,
                'label': labels,
                'patient_id': sample['patient_id'],
                'frame_num': sample['frame_num']
            }
            if self.transform:
                sample_dict = self.transform(sample_dict)
            return sample_dict