import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

x=np.load('image.npz')['arr_0']
y=pd.read_csv("labels.csv"["labels"])
print(pd.Series(y).value_counts())
classes=['A','B','C','D','E']
nclasses=len(classes)
xrain,xtest,train,ytest=train_test_split(x,y,random_state=9,train_size=3500,test_size=500)

def get_prediction(image):
    im_PIL=Image.open(image)
    imagebw=im_PIL.convert("L")
    imagebwResized=imagebw.resize((28,28),Image.ANTIALIAS)W
    pixelFilter=20
    minPixel=np.percentile(imagebwResize,pixelFilter)
    imageInverted=np.clip(imagebwResize-minPixel,0,255)
    maxPixel=np.max(imagebwResize)
    imageInverted=np.asarray(imageInverted)/maxPixel
    testSample=np.array(imageInverted).reshape(1,784)
    testPredict=lr=predict(testSample)
    return testPredict[0]


