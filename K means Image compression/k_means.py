import numpy as np
import matplotlib.pyplot as plt
from Kmeans import KMeans
from PIL import Image
import cv2



def load_img(path):
    img = Image.open(path)
    img = np.array(img)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def preprocess(img):
    #Reshape to a 2D mx3 dataset, i.e three channels per pixel
    img = img.reshape((img.shape[0]*img.shape[1], 3))
    return img


def compress(img, nb_clusters):
    kmeans = KMeans(nb_clusters = nb_clusters)
    kmeans.fit(img)
    y=kmeans.transform(img)
    return y, kmeans

def main():
    img= load_img(r"ped.jpg")
    img = cv2.resize(img, (128, 100), interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("", img); cv2.waitKey(27)
    img = preprocess(img)
    img, kmeans = compress(img, nb_clusters = 5)
    #again reshape to image dimensions with RGB channels
    img = img.reshape((100, 128,3))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    
main()