import pandas as pd
from dataset.image_loader import ImageDataset
from dataset.image_split import get_split_indices, img_dataloader
import torch
from models.resnet18 import load_model
from torch_lr_finder import LRFinder
from engine.engine import engine

#Binary image classification
#new changes
#Read the csv file
df = pd.read_csv('PATH TO THE CSV')

#set the target column and target datasets (Train, Test and validation)
df["target"] = df["updated_shows_scale"] == True
print(df["target"].value_counts())
train_idxs,val_idxs,test_idxs = get_split_indices(df,"target")
df["ds_type"] = df["target"]
df.iloc[train_idxs,-1] = "train"
df.iloc[val_idxs,-1]= "val"
df.iloc[test_idxs,-1] = "test"
print(df["ds_type"].value_counts())

#Load the train and validation datasets.
train_ds = ImageDataset(df,"target","ds_type")
val_ds = ImageDataset(df,"target","ds_type",dt_type="val")
train_dl,val_dl =  img_dataloader(train_ds,val_ds,bs_train = 32,bs_val = 32,upsampling=False)

#select the device.
device= "cuda" if torch.cuda.is_available() else "cpu"
#Load the model
model  = load_model(True,1,device = device)
#print(device)
model = model.to(device)

#Set the optimizer, criterion (here we are doing it for the binary image classification)
optimizer = torch.optim.Adam(model.parameters()) 
criterion = torch.nn.BCELoss()


#Code for finding the Learning Rate.
optimizer = torch.optim.Adam(model.parameters(),lr=1e-7,weight_decay = 1e-2) 
criterion = torch.nn.BCELoss()
lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
#lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_dl, end_lr=100, num_iter=100)

#train the model, using the learning rate from the LRFinder.
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay = 1e-2) 
criterion = torch.nn.BCELoss()
engine(model,15,criterion,optimizer,train_dl,val_dl,device,model_save_folder = "model_checkpoints")