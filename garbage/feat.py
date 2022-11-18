import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np

img=cv2.imread('images/pic_aircraft_a350.jpg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift=cv2.SIFT_create()
orb=cv2.ORB_create()
akaze=cv2.AKAZE_create()

kp=sift.detect(img,None)
print(len(kp))
#kp=np.random.choice(kp,10,False)
okp=orb.detect(img,None)
akp=akaze.detect(img,None)
#des=sift.compute(img)
print(img.shape)


siftimg=cv2.drawKeypoints(img,kp,None,4)
orbimg=cv2.drawKeypoints(img,okp,None,4)
akazeimg=cv2.drawKeypoints(img,akp,None,4)
orbakazeimg=cv2.drawKeypoints(img,okp+akp,None,4)


cv2.imshow('SIFT',siftimg)
cv2.imshow('ORB',orbimg)
cv2.imshow('AKAZE',akazeimg)
cv2.imshow('ORB + AKAZE',orbakazeimg)


cv2.waitKey()
cv2.destroyAllWindows()
print(kp[1])
#sss=cv2.resize(img,dsize=None,fx=0.5,fy=0.5)
#plt.imshow(sss)
#plt.show()
