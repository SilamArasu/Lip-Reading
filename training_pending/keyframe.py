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

def check_dup(path1, path2):
    l1 = path1.split('\n')
    l2 = path2.split('\n')
    
    if len(path1) == 0:
        return
    if len(path2) == 0:
        return
    
    # print("Size of path1 ",len(path1))
    # print("Size of path2 ",len(path2))
    
    for path in l2:
        if path in l1 and path != "":
            print("cmdlist = ")
            print(path1)
            print("new path = ")
            print(path)
            print("-----------------------------------------------------")
            input()

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
    # print("diff_dict before cleaning")
    # pprint(diff_dict)
    # print("Dictionary")
    # print(diff_dict)
    # diff_dict = [k for k,v in diff_dict.items() if v >= thresh]
    # names = [ (x.replace('.png','')).split('-') for x in diff_dict ]
    # key_pics = list()
    # for nam in names:
    #     key_pics.extend(nam)
    # key_pics = list(set(key_pics))
    diff_dict_copy = diff_dict.copy()
    while file_list(diff_dict_copy) > 20 :
        threshold += 100
        # print('threshold is ',threshold)
        diff_dict_copy = {k:v for k,v in diff_dict_copy.items() if v >= threshold}
    
    diff_dict = [k for k,v in diff_dict_copy.items() ]
    # print("diff_dict before cleaning")
    # pprint(diff_dict)
    names = [ (x.replace('.png','')).split('-') for x in diff_dict ]
    key_pics = list()
    for nam in names:
        key_pics.extend(nam)
    key_pics = list(set(key_pics))
    
    # print("unique files")
    # print(key_pics)
    # print('path ',path)
    cmds = ""
    for i in range(len(key_pics)):
         cmds += 'cp "{}" "{}"'.format( os.path.join(path,key_pics[i]+'.png'), os.path.join(dest1,key_pics[i]+'.png')) + '\n'
    return cmds

# obj1 = os.scandir(mapping_path)

er = ""

# for entry1 in obj1 :
#     if entry1.is_dir():

entry1 = sys.argv[1]
print("Your folder is ",entry1)
# entry1 = str(input("Enter the word\t"))
# obj2 = os.scandir(os.path.join(mapping_path, entry1))
obj2 = glob.glob(os.path.join(mapping_path, entry1, '*')) 
cmds = "" 
cmd1 = None
for entry2 in obj2 :
   try:
        cmd1 = None
        # print("Now processing ",os.path.join(mapping_path, entry1, entry2))
        cmd1 = dif_util(os.path.join(mapping_path, entry1, entry2))
        # check_dup(cmds, cmd1)
        cmds += cmd1
        # input()
   except Exception as e:
        er += str(e) + '\n'
            
with open('/home/arasu/FYP/code/sample/keyframe_cmds.sh','w') as f:
    f.write(cmds)
    
with open('/home/arasu/FYP/code/sample/errors.txt','w') as f:
    f.write(er)

print("Time for Execution --- %s seconds ---" % (time.time() - start_time))
print("Reminder that please execute keyframe_cmds.sh")

# # Severe bug in this code
# Linking to non existen images
# check keyframe_cmds.sh. its repeating the same thing

