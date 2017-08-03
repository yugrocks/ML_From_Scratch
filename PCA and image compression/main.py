import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pca import PCA
from PIL import Image
import os
import cv2

def load_imgs():
    #load every image, convert to array and resize
    dataset = []
    for image in os.listdir("images"):
        img = Image.open("images/"+image)
        img = np.array(img)
        #resize to 30x30
        img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
        img = cv2.resize(img, (30, 30), interpolation = cv2.INTER_AREA)
        #add to dataset after flattening into a row vector
        dataset.append(img.flatten())
    return np.array(dataset)


"""First testing on some non image data"""

#read data
dataset = pd.read_csv("Wine.csv")

x = dataset.iloc[:, :-1].values


p = PCA(nb_components = 2)
x2 = p.fit_transform(x)
print(p.get_variance_score())

#Visusalization time
plt.scatter(x2[:,0], x2[:, 1])
plt.plot()
p = None

"""now image compression"""

x = load_imgs()
print("The x matrix contains {} images converted to row vectors of length {}".format(x.shape[0], x.shape[1]))

#Now let's convert them to 25x25 size images. So the number of 
#principal components becomes 25*25 = 625

p = PCA(nb_components = 625)
x3 = p.fit_transform(x)
x2 = p.reconstruct() #Reconstruct to visualize
print(p.get_variance_score())

#Display original
new_img = x[3].reshape((30, 30))
cv2.imshow("",new_img)
cv2.waitKey(0)

#Display converted image
new_img = x2[3].reshape((30, 30))
#new_img = cv2.resize(new_img, (30,30), interpolation = cv2.INTER_AREA)
cv2.imshow("", new_img)
