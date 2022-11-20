import os
import torch
from engine.train_function import fit_one_epoch
from engine.val_function import val_one_epoch
import json
from utils.loss_history import plot_loss_history


def engine(model,num_epochs,criterion,optim,train_dl,val_dl,device,model_save_folder = "model_checkpoints"):

    loss_history=[]
    
    model = model.to(device)
    
    if not os.path.exists(model_save_folder):
        os.mkdir(model_save_folder)
    torch.save(model.state_dict(),f"{model_save_folder}/start.pth")
    
    best_loss = float('inf')    
    
    for epoch in range(num_epochs):
        loss_history_ =  {
          "epoch":epoch,
          "train": None,
          "val": None,
        }        
        print('Epoch {}/{}'.format(epoch + 1, num_epochs ))   

        train_loss,train_accuracy=fit_one_epoch(train_dl,model,criterion,device,optim,epoch,num_epochs)

        loss_history_["train_loss"] = train_loss.item()
        loss_history_["train_acc"] = train_accuracy.item()
        val_loss,val_accuracy = val_one_epoch(val_dl,model,criterion,device)
       
        loss_history_["val_loss"]= val_loss.item()
        loss_history_["val_acc"] = val_accuracy.item()
        loss_history.append(loss_history_)
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            torch.save(model.state_dict(),f'{model_save_folder}/updated_best_model_till_epoch_no_{epoch}.pth')
            try:
                torch.save(model.state_dict(),f'/content/drive/MyDrive/super_ai_data/models/updated_best_model_scale.pth')
            except:
                print("failed to copy to drive")
        
        
        print("train loss: %.6f " %(train_loss.item()))
        print("val loss: %.6f " %(val_loss.item()))
        print("-"*15)
        with open(f"train_details.json","w") as fp:
          json.dump(loss_history,fp,indent=2)

    plot_loss_history("train_details.json")
    return  model, loss_history