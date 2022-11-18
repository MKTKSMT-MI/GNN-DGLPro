import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np

#img=cv2.imread('images/pic_aircraft_a350.jpg')

img=cv2.imread('images/r_005.png',-1)
img=cv2.imread('images/r_2.png',-1)
img=cv2.imread('images/r_002.png',-1)
index=np.where(img[:,:,3]==0)
img[index]=[255,255,255,255]


img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift=cv2.SIFT_create()
orb=cv2.ORB_create()
akaze=cv2.AKAZE_create()

kp,siftdec=sift.detectAndCompute(img,None)
print(len(kp))
#kp=np.random.choice(kp,10,False)
okp=orb.detect(img,None)
akp=akaze.detect(img,None)
#des=sift.compute(img)
print(img.shape)

imgs=[]
siftimg=cv2.drawKeypoints(img,kp,None,4)
imgs.append(siftimg)
orbimg=cv2.drawKeypoints(img,okp,None,4)
imgs.append(orbimg)
akazeimg=cv2.drawKeypoints(img,akp,None,4)
imgs.append(akazeimg)
orbakazeimg=cv2.drawKeypoints(img,okp+akp,None,4)
imgs.append(orbakazeimg)

names=['SIFT','ORB','AKAZE','ORB + AKAZE']

cv2.imshow('SIFT',siftimg)
cv2.imshow('ORB',orbimg)
cv2.imshow('AKAZE',akazeimg)
cv2.imshow('ORB + AKAZE',orbakazeimg)




cv2.waitKey()
cv2.destroyAllWindows()
for i in range(10):
    print(kp[i].response)
fig=plt.figure()
for IMG in range(4):
    fig.add_subplot(2,2,IMG+1)
    plt.imshow(imgs[IMG],cmap='gray')
    plt.title(names[IMG])

#plt.imshow(siftimg,cmap='gray')
#plt.show()
plt.tight_layout()
plt.show()
#sss=cv2.resize(img,dsize=None,fx=0.5,fy=0.5)
#plt.imshow(sss)
#plt.show()
