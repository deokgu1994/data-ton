from .randAugment import RandAugment
import ttach as tta
from torchvision import transforms
import albumentations
# from albumentations import *
import albumentations as A
from albumentations.pytorch import ToTensorV2

# set_tuple = lambda x : tuple([float(z) for z in x.split(",")])
class baseTransform_basic(object):
    def __init__(self, mean, std, resize, use_rand_aug=False):
        self.mean = mean
        self.std = std
        if type(resize) == list:
            self.x, self.y = resize  # notuse
        else:
            self.x, self.y = (512, 512)
        self.use_rand_aug = use_rand_aug
        self.get_transforms()

    def get_transforms(self, need=("train", "val", "eavl")):
        self.transforms = {}
        if "train" in need:
            self.transforms["train"] = transforms.Compose(
                [
                    # transforms resize check 
                    # transforms.Resize((self.x,self.y)),
                    # transforms.CenterCrop((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std),
                ]
            )
            # if self.use_rand_aug:
            #     self.transforms["train"].transforms.insert(1, RandAugment())

        if "val" in need:
            self.transforms["val"] = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std),
                ]
            )
        if "eval" in need:
            self.transforms["eval"] = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    # tta.Rotate90(angles=[0, 90]),
                    # tta.Scale(scales=[1, 2]),
                    tta.FiveCrops(320, 160),
                    # tta.Multiply(factors=[0.7, 1]),
                ]
            )
        return self.transforms

class baseTransformforDataton(object):
    def __init__(self, mean, std, resize, use_rand_aug = False):
        self.mean = mean
        self.std = std
        if type(resize) == list:
            self.x, self.y = resize  # notuse
        else:
            self.x, self.y = (512, 512)
        self.use_rand_aug = use_rand_aug
        self.get_transforms()

    def get_transforms(self,need=('train', 'val', "eavl")):
        self.transforms = {}
        if 'train' in need:
            self.transforms['train'] = albumentations.Compose([
                                                albumentations.augmentations.transforms.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
                                                # albumentations.augmentations.transforms.ToGray(p = 0.5),
                                                # A.Solarize(p = 0.5),
                                                albumentations.augmentations.geometric.rotate.Rotate (limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
                                                A.HorizontalFlip(p=0.5),
                                                A.VerticalFlip(p=0.5),
                                                A.InvertImg(p =0.5),
                                                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255.0, p=1.0),
                                                ToTensorV2(p=1.0),
                                            ], p=1.0)
            # if self.use_rand_aug:
            #     self.transformations["train"].transforms.insert(1, RandAugment())
        if 'val' in need:
            self.transforms['val'] = albumentations.Compose([
                                                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255.0, p=1.0),
                                                ToTensorV2(p=1.0),
                                            ], p=1.0)



class baseTransform(object):
    def __init__(self, mean, std, resize, use_rand_aug=False):
        self.mean = mean
        self.std = std
        if type(resize) == list:
            self.x, self.y = resize  # notuse
        else:
            self.x, self.y = (512, 512)
        self.use_rand_aug = use_rand_aug
        self.get_transforms()

    def get_transforms(self, need=("train", "val", "eavl")):
        self.transforms = {}
        if "train" in need:
            self.transforms["train"] = transforms.Compose(
                [
                    # transforms resize check 
                    transforms.Resize((self.x,self.y)),
                    # transforms.CenterCrop((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std),
                ]
            )
            if self.use_rand_aug:
                self.transforms["train"].transforms.insert(1, RandAugment())

        if "val" in need:
            self.transforms["val"] = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std),
                ]
            )
        if "eval" in need:
            self.transforms["eval"] = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    # tta.Rotate90(angles=[0, 90]),
                    # tta.Scale(scales=[1, 2]),
                    tta.FiveCrops(320, 160),
                    # tta.Multiply(factors=[0.7, 1]),
                ]
            )
        return self.transforms


class customTransform(object):
    def __init__(self, mean, std, resize, use_rand_aug = False):
        self.mean = mean
        self.std = std
        if len(resize) == 1:
            self.x, self.y = (512, 512)
        else:
            self.x, self.y = resize
        self.use_rand_aug = use_rand_aug
        self.get_transforms()
    def get_transforms(self,need=('train', 'val', "eavl")):
        self.transformations = {}
        if 'train' in need:
            self.transformations['train'] = albumentations.Compose([
                                                CenterCrop(height = self.x, width = self.y), # add centercrop 350/350 -> 400/200 -> 300/300
                                                RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                                                Normalize(mean=self.mean, std=self.std, max_pixel_value=255.0, p=1.0),
                                                ToTensorV2(p=1.0),
                                            ], p=1.0)
            # if self.use_rand_aug:
            #     self.transformations["train"].transforms.insert(1, RandAugment())
        if 'val' in need:
            self.transformations['val'] = albumentations.Compose([
                                                CenterCrop(height = self.x, width = self.y), # add centercrop 350/350 -> 400/200 -> 300/300
                                                Normalize(mean=self.mean, std=self.std, max_pixel_value=255.0, p=1.0),
                                                ToTensorV2(p=1.0),
                                            ], p=1.0)


class CustomTransForm_seg(object):
    def __init__(self, mean, std, resize, use_rand_aug=False):
        self.mean = mean
        self.std = std
        self.resize = resize # Not use
        self.use_rand_aug = use_rand_aug
        self.get_transforms()

    def get_transforms(self, need=("train", "val", "eavl")):
        self.transforms = {}
        if "train" in need:
            self.transforms["train"] = A.Compose(
                [
                    # A.GridDropout(holes_number_x=30, holes_number_y=30, p=1.0),
                    # A.Normalize(mean=self.mean, std=self.std, p=1.0),
                    ToTensorV2(p=1.0),
                ]
            )

        if "val" in need:
            self.transforms["val"] = albumentations.Compose(
                [
                    # A.Normalize(mean=self.mean, std=self.std, p=1.0),
                    ToTensorV2(p=1.0),
                ]
            )
