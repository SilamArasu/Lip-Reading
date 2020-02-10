#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 20:46:43 2020

@author: arasu
"""

"""
first and last frame are included.
find no of files in each vid for each word. Least count is the frames to be taken

"""


import cv2
import numpy as np
import os

path = "/home/arasu/FYP/code/sample/differences/four/bbaf4p"
src = "/home/arasu/FYP/code/sample/mapping/four/bbaf4p"
dest = "/home/arasu/FYP/code/sample/keyframes"
thresh = 2000 # You may need to adjust this threshold

files = [ os.path.join(path,x) for x in sorted(os.listdir(path)) if x.endswith('.png')]

# cap = cv2.VideoCapture(video_path)
# # Read the first frame.
# ret, prev_frame = cap.read()

# while ret:
#     ret, curr_frame = cap.read()

#     if ret:

# prev_frame = files[0]
# for index in range(1,len(files)):
#     curr_frame = files[index]
#     diff = cv2.absdiff(cv2.imread(curr_frame,0), cv2.imread(prev_frame,0))
#     non_zero_count = np.count_nonzero(diff)
#     if non_zero_count > p_frame_thresh:
#         print("{} - {}".format(prev_frame,curr_frame))
#     prev_frame = curr_frame

diff_dict = dict()
for f in files:
    diff = np.count_nonzero(cv2.imread(f,0))
    diff_dict[f] = diff
    
diff_dict = {k: v for k, v in sorted(diff_dict.items(), key=lambda item: item[1])}
diff_dict = [k for k,v in diff_dict.items() if v >= thresh]

combined_names = [ (x.split(r'/')[-1]).replace('.png','') for x in diff_dict ]
names = list()
for nam in combined_names:
    names.extend(nam.split('-'))
names = set(names)

src_pics = [ os.path.join(path,x+'.png') for x in names ]
abs_names = [ os.path.join(dest,src.split(r'/')[-2],x+'.png') for x in names ]
abs_names


['/home/arasu/FYP/code/sample/keyframes/four/mouth_035.png',
 '/home/arasu/FYP/code/sample/keyframes/four/mouth_034.png',
 '/home/arasu/FYP/code/sample/keyframes/four/mouth_036.png']