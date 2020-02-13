import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import numpy as np
import os
# import imutils
# import dlib # run "pip install dlib"
# import cv2 # run "pip install opencv-python"
# import glob
import random
import imageio
import gc 
import sys
# from imutils import face_utils
people = ['F01','F02','F04','F05','F06']
data_types = ['words']
folder_enum = ['01','02','03']
instances = ['01','02','03','04','05','06','07','08', '09', '10']
from skimage.transform import resize
max_seq_length = 20
import glob

X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test = []


MAX_WIDTH = 100
MAX_HEIGHT = 50

args = sys.argv[1:]
# args = args.split()
start = int(args[0])
end = int(args[1])

words = ['again']

folder = '/home/arasu/FYP/code/sample'
directory = '/home/arasu/FYP/code/sample/keyframes'
#for person_id in people: # F01, F02, F03, F04, F05
#    instance_index = 0
#    for data_type in data_types: # words

# Variables
# @profile
def my_func():
    word_folder = None
    instances = None
    UNSEEN_TEST_SPLIT = None
    UNSEEN_VALIDATION_SPLIT = None
    i  = None
    path = None
    filelist = None
    sequence = None
    image = None
    pad_array = None
    en = end
    
    for word_index, word in enumerate(words): 
    # folder enum = ['01','02','03']
    # When enumerated - (0,'01'), (1, '02'), (2, '03')
        word_folder = os.path.join(directory, word)
        if len(os.listdir(word_folder)) < en:
                en = len(os.listdir(word_folder))
        instances = os.listdir(word_folder)[start:en]
        random.shuffle(instances)
        UNSEEN_TEST_SPLIT = instances[int(0.6*len(instances)) : int(0.8*len(instances))]
        UNSEEN_VALIDATION_SPLIT = instances[ int(0.8*len(instances)) : ]
        print(f"Word #{word}")
        i = 0 
        for iteration in instances: # ['01','02','03','04','05','06','07','08', '09', '10']
            print("Inside {} {}".format(iteration,str(i)))
            i += 1
            path = os.path.join(directory, word_folder, iteration)
            filelist = sorted(os.listdir(path))
            sequence = []
            image = None 
            for img_name in filelist:
                        image = imageio.imread(os.path.join(path, img_name), as_gray=True)
                        image = resize(image, (MAX_WIDTH, MAX_HEIGHT))
                        # image = np.array(image)
                        # print(image.shape)
                        image = 255 * image
                        # Convert to integer data type pixels.
                        image = image.astype(np.uint8)
                        sequence.append(image)                        
            pad_array = [np.zeros((MAX_WIDTH, MAX_HEIGHT))]                            
            sequence.extend(pad_array * (max_seq_length - len(sequence)))
            sequence = np.array(sequence)
            if iteration in UNSEEN_TEST_SPLIT:
                X_test.append(sequence)
                y_test.append(word_index)
            elif iteration in UNSEEN_VALIDATION_SPLIT:
                X_val.append(sequence)
                y_val.append(word_index)
            else:
                X_train.append(sequence)
                y_train.append(word_index)
        # del sequence
        # del pad_array
        # del image
        # gc.collect()
# #            instance_index += 1
            
            
"""
            # if img_name.startswith('color'):
                image = imageio.imread(os.path.join(path, img_name), as_gray=True)
                image = resize(image, (50, 50))
                # image = np.array(image)
                image = 255 * image
                # Convert to integer data type pixels.
                image = image.astype(np.uint8)
                sequence.append(image)                        
        pad_array = [np.zeros((MAX_WIDTH, MAX_HEIGHT))]                            
        sequence.extend(pad_array * (max_seq_length - len(sequence)))
        sequence = np.array(sequence)
 """   
    
    

    
# print("......")
# print('Finished reading images for person ' + person_id)

# X_train = np.array(X_train)
# X_val = np.array(X_val)
# X_test = np.array(X_test)

if __name__ == '__main__':
    my_func()

"""
X_Train = np.array(X_train)
del X_train
X_Val = np.array(X_val)
del X_val
X_Test = np.array(X_test)
del X_test
"""
np.save('/home/arasu/FYP/code/sample/X_train.npy', np.array(X_train))
np.save('/home/arasu/FYP/code/sample/X_val.npy', np.array(X_val))
np.save('/home/arasu/FYP/code/sample/X_test.npy', np.array(X_test))

"""
y_Train = np.array(y_train)
del y_train
y_Val = np.array(y_val)
del y_val
y_Test = np.array(y_test)
del y_test
"""
np.save('/home/arasu/FYP/code/sample/y_train.npy', np.array(y_train))
np.save('/home/arasu/FYP/code/sample/y_val.npy', np.array(y_val))
np.save('/home/arasu/FYP/code/sample/y_test.npy', np.array(y_test))




