import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset

### 마스크 여부, 성별, 나이를 mapping할 클래스를 생성합니다.
mask_dic = {"incorrect_mask": 1, "mask1":0,"mask2":0,"mask3":0,"mask4":0,"mask5":0, "normal": 2}

class GenderLabels:
    male = 0
    female = 1

class AgeGroup:
    map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) < 58 else 2

class MaskLabels:
    mask = 0
    incorrect = 1
    normal = 2

class MaskBaseDataset(Dataset):
    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []
    def __init__(self, data_dir, mode = "train", transform=None):
        """
        MaskBaseDataset을 initialize 합니다.
        Args:
            img_dir: 학습 이미지 폴더의 root directory 입니다.
            data_dic : {path, image, label}를 저장하는 곳 입니다. 
            transform: Augmentation을 하는 함수입니다.
        """
        self.mode = mode 
        self.img_dir = os.path.join(data_dir, 'images')
        self.data_dic = {}
        label_df = pd.read_csv(os.path.join(data_dir, 'label.csv')) # 라벨 수정된거 
        label_df["file"] = label_df["file"].apply(lambda x: x[x.find("0"):])
        self.label_dic = label_df[["file","class"]].set_index("file").to_dict() # key : file_path, item, class
        self.__ratio = 1

        self.setup()

    def set_transforms(self, transform):
        """
        transform 함수를 설정하는 함수입니다.
        """
        self.transform = transform
        
    def setup(self):
        """
        image의 경로와 각 이미지들의 image, label을 계산하여 저장해두는 함수입니다.
        """
        self._count = 0
        is_limit = False
        for (path, dir, files) in tqdm(os.walk(self.img_dir)):    
            for filename in files:
                if filename[0] == ".": # not in: pass
                    continue
                folder = os.path.split(path)[-1]
                if folder[0] == ".":
                    continue
                gender, age = folder.split("_")[1::2]
                image_path = os.path.join(path, filename)
                mask = filename.split(".")[0]   # ['incorrect_mask', 'jpg']

                label = (mask_dic[mask] * 6) + (getattr(GenderLabels, gender) * 3) + (AgeGroup.map_label(age))

                temp_path = path[path.find("0"):] + "/"+ filename
                try:
                    if label != self.label_dic["class"][temp_path]:
                        # print("!!!"*20, image_path)
                        # print(label , self.label_dic["class"][temp_path])
                        label = self.label_dic["class"][temp_path]
                except KeyError: # added image
                    pass
                self.data_dic[self._count] = {"path": image_path, "image":None, "label":label}
                self._count += 1
            #     if self._count == 100:
            #         is_limit = True
            #         break   
            # if is_limit:
            #      break 

    def __getitem__(self, index):
        # CHECKME: 속도 측면에서 빨라 진다고 생각하고 작성 되었으나, 정말?
        if self.data_dic[index]["image"]== None :
            _image = Image.open(self.data_dic[index]["path"])
            self.data_dic[index]["image"]= _image
        else:
            _image = self.data_dic[index]["image"]  # Image.open

        if self.mode == "train":
            image_transform = self.transform["train"](_image)
        elif self.mode == "val":
            image_transform = self.transform["val"](_image)
        return image_transform, torch.tensor(self.data_dic[index]["label"])

    @property
    def ratio(self,):
        return self.__ratio
    @ratio.setter
    def ratio(self, ratio):
        self.__ratio =  ratio

    def __len__(self):
        return int(len(self.data_dic)* self.__ratio)
        