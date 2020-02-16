# Used to find difference between two frames
# Timing - around 800 seconds (2 speakers)
# Update time - 180 seconds (2 speakers)
# Give word as command line argument
# This code copies the key frames from extracted lips


import numpy as np
import sys,os
import time
import cv2
from pprint import pprint
import glob
start_time = time.time()


sys.path.append(os.path.realpath('..'))

mapping_path = "/home/arasu/FYP/code/sample/mapping"
diff_path = "/home/arasu/FYP/code/sample/mapping/differences"
desk = "/home/arasu/Desktop"
dest = "/home/arasu/FYP/code/sample/keyframes"
# You may need to adjust this threshold
# threshold = 128 

def find_diff(path1, path2):
    # print("reading path = ",path1)
    img1 = cv2.imread(path1,0)
    img2 = cv2.imread(path2, 0)
    
    cv2_subt = cv2.subtract(img1,img2)  
    
    return cv2_subt

def file_list(diff_dict):
    """

    Parameters
    ----------
    diff_dict : dictionary
        DESCRIPTION.

    Returns
    -------
    TYPE
        len of files in it.

    """
    # diff_dict = [k for k,v in diff_dict.items() if v >= thresh]
    diff_dict = [k for k,v in diff_dict.items() ]
    names = [ (x.replace('.png','')).split('-') for x in diff_dict ]
    key_pics = list()
    for nam in names:
        key_pics.extend(nam)
    key_pics = list(set(key_pics))
    return len(key_pics)
    
def dif_util(path):
    # print("Inside ",path)
    threshold = 1000 
    files = [ os.path.join(path,x) for x in sorted(os.listdir(path)) ]
    i = 0
    dest1 = path.replace('mapping','keyframes')
    first = files[0]
    diff_dict = dict()
    res = None
    if len(os.listdir(path)) > 40:
      for index in range(len(files)-1):
          second = files[index+1]
          # print("Comparing {} and {}".format(first,second))
          name1 = (first.split(r'/')[-1]).split('.')[0]
          name2 = (second.split(r'/')[-1]).split('.')[0]
          filename = name1+'-'+name2+'.png'
          res = find_diff(first, second)
          diff = np.count_nonzero(res)
          diff_dict[filename] = diff
          # filename = 'diff_{}.png'.format(str(i).zfill(2))

          # imageio.imsave(os.path.join(dest,filename),res) 
          # # print(os.path.join(dest,filename))
          first = second
          i += 1
      diff_dict = {k: v for k, v in sorted(diff_dict.items(), key=lambda item: item[1])}
      diff_dict_copy = diff_dict.copy()
      while file_list(diff_dict_copy) > 20 :
          threshold += 100
          # print('threshold is ',threshold)
          diff_dict_copy = {k:v for k,v in diff_dict_copy.items() if v >= threshold}

      diff_dict = [k for k,v in diff_dict_copy.items() ]

      names = [ (x.replace('.png','')).split('-') for x in diff_dict ]
      key_pics = list()
      for nam in names:
          key_pics.extend(nam)
      key_pics = list(set(key_pics))
    else:
      key_pics = [ x for x in sorted(os.listdir(path)) ]
    return key_pics
    cmds = ""
    for i in range(len(key_pics)):
         cmds += 'cp "{}" "{}"'.format( os.path.join(path,key_pics[i]+'.png'), os.path.join(dest1,key_pics[i]+'.png')) + '\n'
    return cmds


er = ""

path = 'absolute path to folder'
print("Your folder is ",path)

original = sorted(os.listdir(path)))

keyframes = dif_util(path)
to_be_deleted = [item for item in original if item not in keyframes]

            
