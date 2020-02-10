# Used to find difference between two frames
# Timing - around 800 seconds (2 speakers)
# Update time - 180 seconds (2 speakers)

import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import numpy as np
import imageio
from skimage import img_as_ubyte
import sys,os
import time
import cv2
start_time = time.time()


sys.path.append(os.path.realpath('..'))

mapping_path = "/home/arasu/FYP/code/sample/mapping"
diff_path = "/home/arasu/FYP/code/sample/mapping/differences"
desk = "/home/arasu/Desktop"
dest = "/home/arasu/FYP/code/sample/keyframes"
thresh = 2000 # You may need to adjust this threshold

# threshold = 128 

# This method is not quite accurate
"""
def find_diff(path1, path2):
    # print("reading path = ",path1)
    img1 = mpimg.imread(path1)
    img2 = mpimg.imread(path2)
    
    # Calculate the absolute difference on each channel separately
    error_r = np.fabs(np.subtract(img2[:,:,0], img1[:,:,0]))
    error_g = np.fabs(np.subtract(img2[:,:,1], img1[:,:,1]))
    error_b = np.fabs(np.subtract(img2[:,:,2], img1[:,:,2]))
    
    # Calculate the maximum error for each pixel
    lum_img = np.maximum(np.maximum(error_r, error_g), error_b)
    return lum_img
"""

def find_diff(path1, path2):
    # print("reading path = ",path1)
    img1 = cv2.imread(path1,0)
    img2 = cv2.imread(path2, 0)
    
    cv2_subt = cv2.subtract(img1,img2)  
    
    return cv2_subt

def dif_util(path):
    files = [ os.path.join(path,x) for x in sorted(os.listdir(path)) if x.endswith('.png')]
    i = 0
    dest = path.replace('mapping','differences')
    first = files[0]
    for index in range(len(files)-1):
        second = files[index+1]
        name1 = (first.split(r'/')[-1]).split('.')[0]
        name2 = (second.split(r'/')[-1]).split('.')[0]
        res = find_diff(first, second)
        # filename = 'diff_{}.png'.format(str(i).zfill(2))
        filename = name1+'-'+name2+'.png'
        imageio.imsave(os.path.join(dest,filename),res) 
        # print(os.path.join(dest,filename))
        first = second
        i += 1

obj1 = os.scandir(mapping_path)

for entry1 in obj1 :
    if entry1.is_dir():
        obj2 = os.scandir(os.path.join(mapping_path, entry1.name))
        for entry2 in obj2 :
            if entry2.is_dir():
                # print("Now processing ",os.path.join(mapping_path, entry1.name, entry2.name))
                dif_util(os.path.join(mapping_path, entry1.name, entry2.name))

print("Time for Execution --- %s seconds ---" % (time.time() - start_time))


# thresfunc working, need to integrate this with that
# first run this code


