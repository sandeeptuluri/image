from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np

def get_split_indices(df,column,stratify = True,val_split=0.2,test_split=0.1):
    if stratify:
        train, test_idxs = train_test_split(df.index, test_size=test_split, stratify=df[column])
        train_idxs,val_idxs = train_test_split(train, test_size=val_split, stratify=df.iloc[train,:][column])
    else:
        train, test_idxs = train_test_split(df.index, test_size=test_split, stratify=df[column])
        train_idxs,val_idxs = train_test_split(train, test_size=val_split, stratify=df.iloc[train,:][column])
    return train_idxs,val_idxs,test_idxs


def img_dataloader(train_ds,val_ds,bs_train = 32,bs_val = 32,upsampling=False):
    
    if upsampling:
        labels_unq, counts = np.unique(train_ds.label,return_counts=True)
        class_weights = [sum(counts)/c for c in counts]
        example_wts = [class_weights[e] for e in train_ds.label]
        sampler = WeightedRandomSampler(example_wts,len(train_ds.label))
        train_dl = DataLoader(train_ds,batch_size=bs_train,sampler=sampler)

    else:
        train_dl = DataLoader(train_ds,batch_size=bs_train,shuffle = True)
    val_dl = DataLoader(val_ds,batch_size=bs_val,shuffle = True)

    return train_dl, val_dl