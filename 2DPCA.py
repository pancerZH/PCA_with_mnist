import numpy as np
from PIL import Image
'''
imgs 是三维的图像矩阵，第一维是图像的个数
'''
def TwoDPCA(imgs,p):
    a,b,c = imgs.shape
    print(a,b,c)
    average = np.zeros((b,c))
    for i in range(a):
        average += imgs[i,:,:]/(a*1.0)
    G_t = np.zeros((c,c))
    for j in range(a):
        img = imgs[j,:,:]
        temp = img-average
        G_t = G_t + np.dot(temp.T,temp)/(a*1.0)
    w,v = np.linalg.eigh(G_t)
    w = w[::-1]
    v = v[::-1]
    print(w)
    for k in range(c):
        alpha = sum(w[:k])*1.0/sum(w)
        if alpha >= p:
            u = v[:,:k]
            break
    return u


def TTwoDPCA(imgs,p):
    u = TwoDPCA(imgs,p)
    a1,b1,c1 = imgs.shape
    img = []
    for i in range(a1):
        temp1 = np.dot(imgs[i,:,:],u)
        img.append(temp1.T)
    img = np.array(img)
    uu = TwoDPCA(img,p)
    return u,uu

if __name__ == '__main__':
    im = Image.open('./bloodborne2.jpg')
    im_grey = im.convert('L')
    # im_grey.save('a.png')
    a, b = np.shape(im_grey)
    data = im_grey.getdata()
    data = np.array(data)
    data2 = data.reshape(1, a, b)
    data2_2DPCA = TwoDPCA(data2, 0.9)
    print(data2_2DPCA.shape)
