from torch.utils.data import DataLoader
from dataset import VioDB
from spatial_transforms import (Compose, ToTensor, Normalize,
                                GroupRandomHorizontalFlip,
                                GroupRandomScaleCenterCrop, GroupScaleCenterCrop)
from temporal_transforms import CenterCrop, RandomCrop
from target_transforms import Label


def build_loaders(config):
    # dataset
    dataset = config.dataset
    sample_size = config.sample_size
    stride = config.stride
    sample_duration = config.sample_duration

    # train set
    crop_method = GroupRandomScaleCenterCrop(size=sample_size)
    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    spatial_transform = Compose(
        [crop_method,
         GroupRandomHorizontalFlip(),
         ToTensor(), norm])
    temporal_transform = RandomCrop(size=sample_duration, stride=stride)
    target_transform = Label()

    train_batch = config.train_batch

    train_data = VioDB('../VioDB/{}_jpg/'.format(dataset),
                       '../VioDB/{}_jpg.json'.format(dataset), 'training',
                       spatial_transform, temporal_transform, target_transform)
    train_loader = DataLoader(train_data,
                              batch_size=train_batch,
                              shuffle=True,
                              num_workers=getattr(config, "num_workers", 4),
                              pin_memory=str(config.device).startswith("cuda"))

    # val set
    crop_method = GroupScaleCenterCrop(size=sample_size)
    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    spatial_transform = Compose([crop_method, ToTensor(), norm])
    temporal_transform = CenterCrop(size=sample_duration, stride=stride)
    target_transform = Label()

    val_batch = config.val_batch

    val_data = VioDB('../VioDB/{}_jpg/'.format(dataset),
                     '../VioDB/{}_jpg.json'.format(dataset), 'validation',
                     spatial_transform, temporal_transform, target_transform)
    val_loader = DataLoader(val_data,
                            batch_size=val_batch,
                            shuffle=False,
                            num_workers=getattr(config, "num_workers", 4),
                            pin_memory=str(config.device).startswith("cuda"))

    return train_loader, val_loader
