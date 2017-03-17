from skimage.util import view_as_windows as viewW
import numpy as np
from scipy.ndimage.filters import convolve
import skimage.io as io

def skeletonize(bimg, term_thres=0):
    assert len(bimg.shape) == 2, 'must be binary img'
    def A(a):  
        # woodoo magic !!
        side_size = (3,3)
        ext_size = (side_size[0]-1)//2, (side_size[1]-1)//2
        img = np.pad(a, ([ext_size[0]],[ext_size[1]]), 'constant', constant_values=(0))
        out = viewW(img, side_size)
        out = out.reshape(out.shape[0:2] + (9,))
        out = out[:,:,np.uint8([1,2,5,8,7,6,3,0,1])]
        out[:,:,-1] = out [:,:,0]
        n_0to1 = np.zeros(a.shape, np.uint8)
        n_0to1[:,:] = np.sum(np.diff(out[:,:], axis=2) == 1 , axis=2)
        n_0to1[0,:] = 0
        n_0to1[-1,:] = 0
        n_0to1[:,0] = 0
        n_0to1[:,-1] = 0
        return n_0to1

    n_step1, n_step2 = np.inf ,np.inf

    zthinning = bimg.astype(np.float32)
    zthinning[zthinning !=0] = 1
    
    while n_step1 > term_thres or n_step2 > term_thres:

        step10 = zthinning == 1

        step11_kernel = np.ones((3,3),np.float32)
        step11_mask = convolve(zthinning,step11_kernel)
        step11 = (step11_mask >= 2) & (step11_mask <= 6)


        step12 = A(zthinning) == 1

        step13_kernel = np.float32([[0,-1,0],[0,0,-1],[0,-1,0]])
        step13_mask = convolve(zthinning,step13_kernel)
        step13 = step13_mask > -3

        step14_kernel = np.float32([[0,0,0],[-1,0,-1],[0,-1,0]])
        step14_mask =convolve(zthinning,step14_kernel)
        step14 = step14_mask > -3

        step1_mask = step10 & step11 & step12 & step13 & step14

        n_step1 = np.count_nonzero(step1_mask)

        zthinning[step1_mask] = 0


        step20 = zthinning == 1

        step21_kernel = np.ones((3,3),np.float32)
        step21_mask = convolve(zthinning,step21_kernel)
        step21 = (step21_mask >= 2) & (step21_mask <= 6)

        step22 = A(zthinning) == 1

        step23_kernel = np.float32([[0,-1,0],[-1,0,-1],[0,0,0]])
        step23_mask = convolve(zthinning,step23_kernel)
        step23 = step23_mask > -3

        step24_kernel = np.float32([[0,-1,0],[-1,0,0],[0,-1,0]])
        step24_mask = convolve(zthinning,step24_kernel)
        step24 = step24_mask > -3

        step2_mask = step20 & step21 & step22 & step23 & step24

        n_step2 = np.count_nonzero(step2_mask)
        zthinning[step2_mask] = 0
    
    return zthinning.astype(np.uint8)
