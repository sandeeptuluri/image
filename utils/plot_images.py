import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random

df = pd.read_csv('PATH TO THE CSV')

def plot_images(df,cat,cat_value):
  
    img_names_tr = list(df[df[cat] == cat_value].image_url)
    img_names_false = list(df[df[cat] != cat_value].image_url)
    random.shuffle(img_names_tr)
    random.shuffle(img_names_false)
    n_rows = 3
    r_cols = 4
    plt.figure(figsize=(20,16))
    for i in range(n_rows*r_cols):
        plt.subplot(n_rows,r_cols,i+1)
        if i%2 == 0:
            img_name = img_names_tr[i].split('/')[-1]
            img = cv2.imread(f"images/{img_name}.jpg")
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            plt.title(f"{cat_value} \n{img_name[5:]}")
            plt.imshow(img)
        else:

            img_name = img_names_false[i].split('/')[-1]
            img = cv2.imread(f"images/{img_name}.jpg")
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            plt.title(f"Not {cat_value} \n{img_name[5:]}")
            plt.imshow(img)
    plt.axis("off") 


if __name__ == '__main__':
    plot_images(df,'one of the column',True)