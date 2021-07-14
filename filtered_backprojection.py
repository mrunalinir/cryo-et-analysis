import mrcfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from scipy.fftpack import fft, fftshift, ifft
from scipy.ndimage import zoom
from tqdm.auto import tqdm
import cv2 


file1 = "10453/10453/10453/data/tilt_series/TS_005.mrc"
with mrcfile.open(file1, mode='r+', permissive=True) as mrc:
     mrc.header.map = mrcfile.constants.MAP_ID
k=mrcfile.open(file1,mode='r+',permissive=True)
# k1=mrcfile.open('reconstruction.mrc',mode='r+',permissive=True)

print(k.data.shape)
# , k1.data.shape)


def projFilter(sino):
    """
    inputs: sino - [n x m] numpy array where n is the number of projections and m is the number of angles used.
    outputs: filtSino - [n x m] filtered subtomogram array"""
    
    a = 0.1; 
    projLen, numAngles = sino.shape
    step = 2*np.pi/projLen
    w = arange2(-np.pi, np.pi, step)
    if len(w) < projLen:
        w = np.concatenate([w, [w[-1]+step]]) #depending on image size, it might be that len(w) =  
                                              #projLen - 1. Another element is added to w in this case
    rn1 = abs(2/a*np.sin(a*w/2));  #approximation of ramp filter abs(w) with a funciton abs(sin(w))
    rn2 = np.sin(a*w/2)/(a*w/2);   #sinc window with 'a' modifying the cutoff freqs
    r = rn1*(rn2)**2;              #modulation of ramp filter with sinc window
    
    filt = fftshift(r)   
    filtSino = np.zeros((projLen, numAngles))
    for i in range(numAngles):
        projfft = fft(sino[:,i])
        filtProj = projfft*filt
        filtSino[:,i] = np.real(ifft(filtProj))

    return filtSino

def backproject(sinogram, theta):
    """Backprojection function. 
    inputs:  sinogram - [n x m] numpy array where n is the number of projections and m the number of angles
             theta - vector of length m denoting the angles represented in the sinogram
    output: backprojArray - [n x n] backprojected 2-D numpy array"""
    imageLen = sinogram.shape[0]
    reconMatrix = np.zeros((imageLen, imageLen))
    
    x = np.arange(imageLen)-imageLen/2 #create coordinate system centered at (x,y = 0,0)
    y = x.copy()
    X, Y = np.meshgrid(x, y)

    plt.ion()
    fig2, ax = plt.subplots()
    im = plt.imshow(reconMatrix, cmap='gray')

    theta = theta*np.pi/180
    numAngles = len(theta)

    for n in range(numAngles):
        Xrot = X*np.sin(theta[n])-Y*np.cos(theta[n]) #determine rotated x-coordinate about origin in mesh grid form
        XrotCor = np.round(Xrot+imageLen/2) #shift back to original image coordinates, round values to make indices
        XrotCor = XrotCor.astype('int')
        projMatrix = np.zeros((imageLen, imageLen))
        m0, m1 = np.where((XrotCor >= 0) & (XrotCor <= (imageLen-1))) #after rotating, you'll inevitably have new coordinates that exceed the size of the original
        s = sinogram[:,n] #get projection
        projMatrix[m0, m1] = s[XrotCor[m0, m1]]  #backproject in-bounds data
        reconMatrix += projMatrix
        im.set_data(Image.fromarray((reconMatrix-np.min(reconMatrix))/np.ptp(reconMatrix)*255))
        ax.set_title('Theta = %.2f degrees' % (theta[n]*180/np.pi))
        fig2.canvas.draw()
        fig2.canvas.flush_events()

    plt.close()
    plt.ioff()
    backprojArray = np.flipud(reconMatrix)
    return backprojArray



dTheta = 3
theta = np.arange(-60,60,dTheta)

filtSino=k.data
s_=32
filtSin=np.zeros((41,s_,s_))
for i in range(len(filtSino)):
    filtSin[i,:,:]=cv2.resize(filtSino[i],(s_,s_))
print(filtSin.shape)
filtSino=filtSin
orig=np.zeros((s_,s_,s_))
for i in tqdm(range((filtSino.shape[2]))):
    filt=filtSino[:,:,i].transpose(1,0)
    recon = backproject(filt, theta)
    recon2 = np.round((recon-np.min(recon))/np.ptp(recon)*255) #convert values to integers 0-255
    orig[:,:,i]=recon
print(orig.shape)

ar=np.zeros(orig.shape)
ar1=orig[10:19]
c=0
for i in range(len(ar)):
    print(c,i)
    
    if i%4==0:
        ar[i]=ar1[c]
    elif i%4 ==1:
        ar[i]=(3*ar1[c]+ar1[c+1])/4
    elif i%4==2:
        ar[i]=(2*ar1[c]+2*ar1[c+1])/4
    elif i%4==3:
        ar[i]=(ar1[c]+3*ar1[c+1])/4
        c+=1
print(ar.shape)
    
img=ar
fig, ax = plt.subplots(nrows=8, ncols=4,figsize=(12,12))
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
plt.margins(0,0)
i=0
for row in ax:
    for col in row:
        f=img[i]
        # f-=np.min(f)
        # f/=np.max(f)
        # f*=255
        col.imshow(f)
        i+=1
plt.savefig("tomo.jpg")

# orig=k1.data
# print(orig.shape)
# orig=zoom(orig,(0.125,0.125,0.125))
# orig=zoom(orig,(0.5,0.5,0.5))
# print(orig.shape)
# img=orig[:,:,:]
# import numpy as np
# fig, ax = plt.subplots(nrows=8, ncols=4,figsize=(16,16))
# i=0
# for row in ax:
#     for col in row:
#         f=img[i]
#         f-=np.min(f)
#         f/=np.max(f)
#         f*=255
#         col.imshow(f)
#         i+=1

# plt.savefig("tomo.jpg")

# lab=k1.data
# lab=zoom(lab,(0.125,0.125,0.125))
# lab=zoom(lab,(0.5,0.5,0.5))
# lab-=np.min(lab)
# lab/=np.max(lab)