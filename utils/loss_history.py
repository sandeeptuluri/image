import json
import matplotlib.pyplot as plt


def plot_loss_history(train_history):
    train_history = json.load(open(train_history))
    val_loss = [a['val_loss'] for a in train_history]
    train_loss = [a['train_loss'] for a in train_history]
    train_acc = [a['train_acc'] for a in train_history]
    val_acc = [a['val_acc'] for a in train_history]
    epochs = [a['epoch'] for a in train_history]
    plt.figure(figsize = (8,6))
    plt.plot(epochs, train_loss, label = "train_loss", linestyle="--")
    plt.plot(epochs, val_loss, label = "val_loss", linestyle="--")
    plt.plot(epochs, train_acc, label = "train_acc", linestyle="--")
    plt.plot(epochs, val_acc, label = "val_acc", linestyle="--")
    plt.legend()
    plt.show()