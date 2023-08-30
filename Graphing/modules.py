import numpy as np
import matplotlib.pyplot as plt
import torch

def img2patch(image,patch_num,cmap='gray',views=False):
    if views:        
        if image.shape[0]==3 or image.shape[0]==1:
            c=image.shape[0]
            image=image.permute(1,2,0)
        else:
            c=image.shape[2]

        size=image.shape[0]
        patch_width=int(size/patch_num)
        data=[]

        for i in range(0,size,patch_width):
            for j in range(0,size,patch_width):
                data.append(image[i:i+patch_width,j:j+patch_width,:])

        fig=plt.figure()

        for i in range(patch_num**2):
            ax=fig.add_subplot(patch_num,patch_num,i+1)
            ax.imshow(data[i],cmap=cmap,vmin=0,vmax=255)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

        return data

    else:
        size=image.shape[1]
        patch_width=int(size/patch_num)
        data=[]

        for i in range(0,size,patch_width):
            for j in range(0,size,patch_width):
                data.append(image[:,i:i+patch_width,j:j+patch_width])
        return torch.stack(data)


def objYN(data,patch_num,views=False):
    
    obj_patchs=[]
    for d in data:
        percentage=np.sum(d==255)/d.size
        if percentage<0.1:
            obj_patchs.append(np.full((d.shape[0],d.shape[0]),0))
        else:
            obj_patchs.append(np.full((d.shape[0],d.shape[0]),255))
    
    if views:
        fig=plt.figure()

        for i in range(patch_num**2):
            ax=fig.add_subplot(patch_num,patch_num,i+1)
            ax.imshow(obj_patchs[i],cmap='jet',vmin=0,vmax=255)
            ax.axis('off')

        plt.tight_layout()
        plt.show()
