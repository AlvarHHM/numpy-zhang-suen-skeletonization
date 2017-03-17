# numpy-zhang-suen-skeletonization
Implementation of zhang-suen using numpy
Huge speedup compare to pixel looping implementation in pure python

```python
import zhang_suen
binary_img = io.imread('test_thin.bmp')
from skimage.filters import threshold_otsu
Otsu_Threshold = threshold_otsu(binary_img)   
binary_img = binary_img < 240
plt.imshow(binary_img)
plt.figure()
skeleton = zhang_suen.skeletonize(binary_img) 
plt.imshow(skeleton)
```
[![IMAGE ALT TEXT HERE](https://github.com/AlvarHHM/numpy-zhang-suen-skeletonization/blob/master/result.png?raw=true)]
