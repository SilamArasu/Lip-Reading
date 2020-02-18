#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:09:57 2020

@author: arasu
"""

import os
import fnmatch
import errno
import random
import imageio
import numpy as np
from scipy import ndimage
from scipy.misc import imresize
from skimage import io
import skvideo.io
import dlib
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import time
from skimage.transform import resize
import pickle

MAX_WIDTH = 100
MAX_HEIGHT = 50

face_predictor_path = '/home/arasu/FYP/code/shape_predictor_68_face_landmarks.dat'
predict_path = '/home/arasu/FYP/code/sample/predict'
max_seq_length = 40

class Video(object):
    """
    Class for preprocessing the video
    """
    def __init__(self, vtype, face_predictor_path):
        if vtype == 'face' and face_predictor_path is None:
            raise AttributeError('Please specify the path for shape_predictor_68_face_landmarks.dat file')
        self.face_predictor_path = face_predictor_path
        self.vtype = vtype
        self.face = None   # Contains the frames of faces
        self.mouth = None   # Contains the frames of mouths

    def from_video(self, path):
        """
        Read from videos
        """
        frames = self.get_video_frames(path)
        self.process_frames_face(frames)
        return self

    def process_frames_face(self, frames):
        """
        Preprocess from frames using face detector
        """
        detector = dlib.get_frontal_face_detector() # dlib’s pre-trained face detector
        predictor = dlib.shape_predictor(self.face_predictor_path) # Identifies the facial landmark
        mouth_frames = self.get_frames_mouth(detector, predictor, frames)
        self.face = np.array(frames)
        self.mouth = np.array(mouth_frames)

    def get_frames_mouth(self, detector, predictor, frames):
        """
        Get frames using mouth crop
        """
        mouth_width = 100
        mouth_height = 50
        horizontal_pad = 0.19
        normalize_ratio = None
        mouth_frames = []
        
        for frame in frames:
            dets = detector(frame, 1) # detect faces (detecting the bounding box of faces in the image)
            shape = None
            for det in dets: # for each faces detected
                shape = predictor(frame, det)
                i = -1
            if shape is None: # Predictor can't detect facial landmarks, just return None
                return [None]
            mouth_points = []
            
            # Each part corresponds to each landmark (1-68)
            for part in shape.parts():
                i += 1
                if i < 48: # Only take mouth region
                    continue
                mouth_points.append((part.x, part.y))
                
            np_mouth_points = np.array(mouth_points)
            mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0) # Find the centre of the lips 
            
            # We normalize to maintain the aspect ratio while saving the mouth 
            if normalize_ratio is None:
                # Padding is done because we can't consider only the lips. The mouth too contribute in lip reading
                mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - horizontal_pad)
                mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + horizontal_pad)
                normalize_ratio = mouth_width / float(mouth_right - mouth_left)

            new_img_shape = (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio))
            resized_img = imresize(frame, new_img_shape)
            mouth_centroid_norm = mouth_centroid * normalize_ratio

            mouth_l = int(mouth_centroid_norm[0] - mouth_width / 2)
            mouth_r = int(mouth_centroid_norm[0] + mouth_width / 2)
            mouth_t = int(mouth_centroid_norm[1] - mouth_height / 2)
            mouth_b = int(mouth_centroid_norm[1] + mouth_height / 2)

            mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]
            mouth_frames.append(mouth_crop_image)
            
        return mouth_frames

    def get_video_frames(self, path):
        """
        Get video frames 
        """
        videogen = skvideo.io.vreader(path) # Loads video frame-by-frame
        frames = np.array([frame for frame in videogen])
        return frames


def extract_lips(filepath):
    video = Video(vtype='face', face_predictor_path=face_predictor_path).from_video(filepath)
    
    if video.mouth[0] is None:
        print("Can't detect face")
    
    i = 0
    for frame in video.mouth:
        io.imsave(os.path.join(predict_path, "mouth_{0:03d}.png".format(i)), frame)
        i += 1
        
def prepare_frames():
    filelist = sorted(os.listdir(predict_path))
    sequence = []
    image = None 
    for img_name in filelist:
                image = imageio.imread(os.path.join(predict_path, img_name), as_gray=True)
                image = resize(image, (MAX_WIDTH, MAX_HEIGHT))
                # image = np.array(image)
                # print(image.shape)
                image = 255 * image
                # Convert to integer data type pixels.
                image = image.astype(np.uint8)
                sequence.append(image)                        
    pad_array = [np.zeros((MAX_WIDTH, MAX_HEIGHT), dtype=np.int8) ]                            
    sequence.extend(pad_array * (max_seq_length - len(sequence)))
    return np.array(sequence)
    
def normalize_it(X):
    v_min = X.min(axis=(2, 3), keepdims=True)
    v_max = X.max(axis=(2, 3), keepdims=True)
    X = (X - v_min)/(v_max - v_min)
    X = np.nan_to_num(X)
    return X


    
filepath = '/home/arasu/FYP/code/GridCorpus/s1/brba1a.mpg'
extract_lips(filepath)

sequence = prepare_frames()
predict = []
predict.append(sequence)
predict = np.array(predict)

predict = normalize_it(predict)
predict = np.expand_dims(predict, axis=4)

model = None
with open('/home/arasu/FYP/code/sample/my_model6.pickle', 'rb') as f:
    model = pickle.load(f)
    
ypred = model.predict(predict)