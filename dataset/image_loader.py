import torch
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.io import read_image

img_name_fn = lambda x: f"PATH TO THE IMAGES{x.split('/')[-1]}.jpg"

class ImageDataset(Dataset):

    def __init__(self,df,cat,class_type = "class_type",dt_type="train",img_size = (224,224),transforms_=False):
        mapper = {True:1,False:0}
        df["image_url"].apply(img_name_fn)
        self.dt_type = dt_type
        self.df = df[df[class_type] == self.dt_type]
        self.label = list(self.df[cat].map(mapper).values)
        self.df["img_name"] = self.df["image_url"].apply(img_name_fn)
        self.img_names = list(self.df["img_name"].values)
        self.default_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(size=img_size),transforms.ToTensor()])
        self.transform = self.transforms(transforms_)


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self,idx):
        image = read_image(self.img_names[idx])
        image = self.transform(image)
        label = self.label[idx]
        return image,label
    def transforms(self,transforms):
        if self.dt_type != "test":
            if not transforms:
                return  self.default_transform
            else:
                return transforms
        else:
            return self.default_transform