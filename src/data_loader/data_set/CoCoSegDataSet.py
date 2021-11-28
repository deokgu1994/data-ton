import os
import numpy as np
import cv2
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class CustomSegDataSet(Dataset):
    """COCO format"""
    category_names =['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
    def __init__(self, data_dir, ann_dir, mode="train", transform=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.data_dir = data_dir
        self.coco = COCO(ann_dir)
        self.__ratio = 1

    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.data_dir, image_infos["file_name"]))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0

        if self.mode == "train" or self.mode == "val":
            ann_ids = self.coco.getAnnIds(imgIds=image_infos["id"])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # General trash = 1, ... , Cigarette = 10
            anns = sorted(
                anns, key=lambda idx: len(idx["segmentation"][0]), reverse=False
            )
            for i in range(len(anns)):
                className = self.get_classname(anns[i]["category_id"], cats)
                pixel_value = self.category_names.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)

            if self.mode == "train":
                # transform -> albumentations 라이브러리 활용
                if self.transforms["train"] is not None:
                    transformed = self.transforms["train"](image=images, mask=masks)
                    images = transformed["image"]
                    masks = transformed["mask"]
                return images, masks, image_infos
            elif self.mode == "val":
                # transform -> albumentations 라이브러리 활용
                if self.transforms["val"] is not None:
                    transformed = self.transforms["val"](image=images)
                    images = transformed["image"]
                return images, image_infos

        if self.mode == "eval":
            # transform -> albumentations 라이브러리 활용
            if self.transforms["eval"] is not None:
                transformed = self.transforms["eval"](image=images)
                images = transformed["image"]
            return images, image_infos

    @property
    def ratio(self,):
        return self.__ratio
    @ratio.setter
    def ratio(self, ratio):
        self.__ratio =  ratio

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return int(len(self.coco.getImgIds()) * self.__ratio)
        # return len(self.coco.getImgIds())

    def set_transforms(self, transforms):
        # transform = {"train", "val", "eval"}
        self.transforms = transforms
    
    def get_classname(self, classID, cats):
        for i in range(len(cats)):
            if cats[i]['id']==classID:
                return cats[i]['name']
        return "None"