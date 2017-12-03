import numpy as np
from PIL import Image
'''
imgs 是三维的图像矩阵，第一维是图像的个数
'''
def TwoDPCA(imgs, dim):
    a,b,c = imgs.shape
    average = np.zeros((b,c))
    for i in range(a):
        average += imgs[i,:,:]/(a*1.0)
    G_t = np.zeros((c,c))
    for j in range(a):
        img = imgs[j,:,:]
        temp = img-average
        G_t = G_t + np.dot(temp.T,temp)/(a*1.0)
    w,v = np.linalg.eigh(G_t)
    # print('v_shape:{}'.format(v.shape))
    w = w[::-1]
    v = v[::-1]
    '''
    for k in range(c):
        # alpha = sum(w[:k])*1.0/sum(w)
        alpha = 0
        if alpha >= p:
            u = v[:,:k]
            break
    '''
    u = v[:,:dim]
    return u  # u是投影矩阵


def TTwoDPCA(imgs, dim):
    u = TwoDPCA(imgs, dim)
    a1,b1,c1 = imgs.shape
    img = []
    for i in range(a1):
        temp1 = np.dot(imgs[i,:,:],u)
        img.append(temp1.T)
    img = np.array(img)
    print('img_shape:{}'.format(img.shape))
    uu = TwoDPCA(img, dim)
    print('uu_shape:{}'.format(uu.shape))
    return u,uu  # uu是投影矩阵

if __name__ == '__main__':
    im = Image.open('./bloodborne2.jpg')
    im_grey = im.convert('L')
    # im_grey.save('a.png')
    a, b = np.shape(im_grey)
    data = im_grey.getdata()
    data = np.array(data)
    data2 = data.reshape(1, a, b)
    print('data2_shape:{}'.format(data2.shape))
    data2_2DPCA, data2_2D2DPCA = TTwoDPCA(data2, 10)
    print('data2_2DPCA:{}'.format(data2_2DPCA.shape))
    print('data2_2D2DPCA:{}'.format(data2_2D2DPCA.shape))
