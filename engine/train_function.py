import torch
import numpy as np
from tqdm import tqdm

def fit_one_epoch(train_loader,model,criterion,device,optimizer, epoch, num_epochs,thresh = 0.5 ): 
    step_train = 0

    train_losses = list()
    train_acc = list()
    model.train()
    for i, (images, targets) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        targets = targets.unsqueeze(1).float()
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        train_losses.append(loss.item())

        preds = torch.Tensor(np.where(logits.detach().cpu() < thresh, 0, 1))

        num_correct = sum(preds.eq(targets.detach().cpu()))
        running_train_acc = float(num_correct) / float(images.shape[0])
        train_acc.append(running_train_acc)
        
    train_loss = torch.tensor(train_losses).mean()  
    train_accuracy = torch.tensor(train_acc).mean()   
    print(f'Epoch {epoch}/{num_epochs-1}')  
    print(f'Training loss: {train_loss:.2f}')
    print(f'Training accuracy: {train_accuracy*100:.2f} %') 
    return train_loss,train_accuracy