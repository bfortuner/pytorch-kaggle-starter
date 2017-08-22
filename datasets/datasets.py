import torch
from . import data_utils


loaders = {
    'pil': data_utils.pil_loader,
    'tns': data_utils.tensor_loader,
    'npy': data_utils.numpy_loader,
    'tif': data_utils.tif_loader,
    'io': data_utils.io_loader
}


class FileDataset(torch.utils.data.Dataset):
    def __init__(self, fpaths,
                 img_loader='pil',
                 targets=None,
                 transform=None,
                 target_transform=None):
        self.fpaths = fpaths
        self.loader = self._get_loader(img_loader)
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def _get_loader(self, loader_type):
        return loaders[loader_type]

    def _get_target(self, index):
        if self.targets is None:
            return torch.FloatTensor(1)
        target = self.targets[index]
        if self.target_transform is not None:
            return self.target_transform(target)
        return torch.FloatTensor(target)

    def _get_input(self, index):
        img_path = self.fpaths[index]
        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        input_ = self._get_input(index)
        target = self._get_target(index)
        img_path = self.fpaths[index]
        return input_, target, img_path

    def __len__(self):
        return len(self.fpaths)


class MultiInputDataset(FileDataset):
    def __init__(self, fpaths,
                 img_loader='pil', #'tns', 'npy'
                 targets=None,
                 other_inputs=None,
                 transform=None,
                 target_transform=None):
        super().__init__(fpaths, img_loader, targets,
                         transform, target_transform)
        self.other_inputs = other_inputs

    def _get_other_input(self, index):
        other_input = self.other_inputs[index]
        return other_input

    def __getitem__(self, index):
        input_ = self._get_input(index)
        target = self._get_target(index)
        other_input = self._get_other_input(index)
        img_path = self.fpaths[index]
        return input_, target, other_input, img_path


class MultiTargetDataset(FileDataset):
    def __init__(self, fpaths,
                 img_loader='pil',
                 targets=None,
                 other_targets=None,
                 transform=None,
                 target_transform=None):
        super().__init__(fpaths, img_loader, targets,
                         transform, target_transform)
        self.other_targets = other_targets

    def _get_other_target(self, index):
        if self.other_targets is None:
            return torch.FloatTensor(1)
        other_target = self.other_targets[index]
        return torch.FloatTensor(other_target)

    def __getitem__(self, index):
        input_ = self._get_input(index)
        target = self._get_target(index)
        other_target = self._get_other_target(index)
        img_path = self.fpaths[index]
        return input_, target, other_target, img_path


class ImageTargetDataset(torch.utils.data.Dataset):
    def __init__(self, input_fpaths,
                target_fpaths,
                input_loader='pil',
                target_loader='pil',
                input_transform=None,
                target_transform=None,
                joint_transform=None):
        self.input_fpaths = input_fpaths
        self.target_fpaths = target_fpaths
        self.input_loader = loaders[input_loader]
        self.target_loader = loaders[target_loader]
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

    def _get_target(self, index):
        if self.target_fpaths is None:
            return torch.FloatTensor(1), ""
        img_path = self.target_fpaths[index]
        img = self.target_loader(img_path)
        if self.target_transform is not None:
            img = self.target_transform(img)
        return img, img_path

    def _get_input(self, index):
        img_path = self.input_fpaths[index]
        img = self.input_loader(img_path)
        if self.input_transform is not None:
            img = self.input_transform(img)
        return img, img_path

    def __getitem__(self, index):
        input_, inp_path = self._get_input(index)
        target, tar_path = self._get_target(index)
        if self.joint_transform is not None:
            input_, target = self.joint_transform(input_, target)
        return input_, target, inp_path, tar_path

    def __len__(self):
        return len(self.input_fpaths)
