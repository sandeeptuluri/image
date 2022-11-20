from torchvision.models import resnet18, ResNet18_Weights
import torchvision
import torch
import torch.nn as nn

def load_model(freeze_backbone,out_features,model_path=False):
    
    model = torchvision.models.resnet18(pretrained=True)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, out_features),
        nn.Sigmoid()           
        )
    if model_path:
        model.load_state_dict(torch.load(model_path))
    return model