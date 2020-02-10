# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# ----------------------------------------
# This code segregates the frames to their respective word
# For eg, if "again" is spoken in bbaf4a @ frames 40-54, they will be stored(symlinked) in again/bbaf4a/
# run align_database
# run cmd.sh in datasets folder like
# sh cmd.sh > /dev/null
# dev/null somehow fastens the process 10 times
# Timing - around 80 seconds (2 speakers)
# ----------------------------------------


import pandas as pd
import subprocess 
import os 
import time
import glob
start_time = time.time()

datasets = "/home/arasu/FYP/code/datasets"
vids_path = "/home/arasu/FYP/code/datasets/lips"
align_path = "/home/arasu/FYP/code/sample/bbaf2n.align"
imgs_path = "/home/arasu/FYP/code/sample/bbaf2n"
mapping = "/home/arasu/FYP/code/sample/mapping"
er = ""
cmd = ""
# path = each video folder - bbaf2n folder

# Copies the frames to respective folder
def copy_pics(path, start, end, word):
    folder_name = path.split(r'/')[-1]
    # print("Calling mkdir -p '{}'".format(os.path.join(mapping,word)))
    # print("Calling mkdir -p '{}'".format(os.path.join(mapping,word,folder_name)))
    subprocess.check_output("mkdir -p '{}'".format(os.path.join(mapping,word,folder_name)), shell=True)
    #curr_list = pics_list[start:end+1]
    l = ""
    global er
    for i in range(start,end+1):
        try:
            # subprocess.check_output("ln -s '{}.png' '{}'".format(os.path.join(path,"mouth_0")+str(i).zfill(2), os.path.join(mapping,word,folder_name)+'/'), shell=True)
            # print("cp '{}.png' '{}'".format(os.path.join(path,"mouth_0")+str(i).zfill(2), os.path.join(mapping,word,folder_name)+'/'))
            l += "ln -s '{}.png' '{}'".format(os.path.join(path,"mouth_0")+str(i).zfill(2), os.path.join(mapping,word,folder_name)+'/') + '\n'
        except:
        #    print("Error")
            er += "Error in "+"cp '{}.png' '{}'".format(os.path.join(path,"mouth_0")+str(i).zfill(2), os.path.join(mapping,word,folder_name)+'/') + "\n"

    #print("Done '{}'".format(os.path.join(mapping,word,folder_name)))    
    return l

# Splits the align    
def func(path):
    folder_name = path.split(r'/')[-1]
    align_path = os.path.join(datasets,"align",folder_name)+".align"
    df = pd.read_csv(align_path, delimiter=' ', header=None)
    df.columns = ['Start', 'End', 'Word']
    df['Start'] = (df['Start']/1000).astype(int)
    df['End'] = (df['End']/1000).astype(int)
    
    # p = subprocess.Popen("ls {}".format(imgs_path), stdout=subprocess.PIPE, shell=True)
    # (output, err) = p.communicate()
    # output = output.decode()
    # output = output.split('\n')
    # output = [os.path.join(imgs_path,x) for x in output]
    l = ""
    for index, row in df.iterrows():
        if row['Word'] == 'sil':
            continue
     #   print(row['Start'], row['End'], row['Word'])
        l += copy_pics(path, row['Start'], row['End'], row['Word'])
    return l
        
for path1 in glob.glob(os.path.join(datasets, "lips", '*')):    
    for path2 in glob.glob(os.path.join(path1, '*')): 
        if len(os.listdir(path2)) == 0:
            continue
      #  print(path2)
        cmd += func(path2)   
    
with open(datasets+"/errors.txt", 'w') as f:
    f.write(er)    
with open(datasets+"/cmd.sh", 'w') as f:
    f.write(cmd)  
    
print("Time for Execution --- %s seconds ---" % (time.time() - start_time))
