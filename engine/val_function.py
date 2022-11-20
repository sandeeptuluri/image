import torch
import numpy as np

def val_one_epoch(val_loader,model,criterion,device,thresh = 0.5):
    val_losses = list()
    val_accs = list()
    model.eval()
    step_val = 0
    with torch.no_grad():
        for (images, targets) in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            targets = targets.unsqueeze(1).float()
            loss = criterion(logits, targets)
            val_losses.append(loss.item())      

            preds = torch.Tensor(np.where(logits.detach().cpu() < thresh, 0, 1))
            num_correct = sum(preds.eq(targets.detach().cpu()))
            running_val_acc = float(num_correct) / float(images.shape[0])

            val_accs.append(running_val_acc)
      

    val_loss = torch.tensor(val_losses).mean()
    val_acc = torch.tensor(val_accs).mean() 
  
    print(f'Validation loss: {val_loss:.2f}')  
    print(f'Validation accuracy: {val_acc*100:.2f} %') 
    return val_loss,val_acc