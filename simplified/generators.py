from helpers import text_to_labels
from videos import Video
from aligns import Align
from threadsafe import threadsafe_generator
#from lipnet.helpers.list import get_list_safe
from keras import backend as K
import numpy as np
import keras
import pickle
import os
import glob
import multiprocessing

class BasicGenerator(keras.callbacks.Callback):
    def __init__(self, dataset_path, minibatch_size, img_c, img_w, img_h, frames_n, absolute_max_string_len=30, **kwargs):
        self.dataset_path   = dataset_path
        self.minibatch_size = minibatch_size
        self.blank_label    = self.get_output_size() - 1
        self.img_c          = img_c
        self.img_w          = img_w
        self.img_h          = img_h
        self.frames_n       = frames_n
        self.absolute_max_string_len = absolute_max_string_len
        self.cur_train_index = multiprocessing.Value('i', 0)
        self.cur_val_index   = multiprocessing.Value('i', 0)
        self.curriculum      = kwargs.get('curriculum', None)
        self.random_seed     = kwargs.get('random_seed', 13)
        self.vtype               = kwargs.get('vtype', 'mouth')
        self.face_predictor_path = kwargs.get('face_predictor_path', None)
        self.steps_per_epoch     = kwargs.get('steps_per_epoch', None)
        self.validation_steps    = kwargs.get('validation_steps', None)
        # Process epoch is used by non-training generator (e.g: validation)
        # because each epoch uses different validation data enqueuer
        # Will be updated on epoch_begin
        self.process_epoch       = -1
        # Maintain separate process train epoch because fit_generator only use
        # one enqueuer for the entire training, training enqueuer can contain
        # max_q_size batch data ahead than the current batch data which might be
        # in the epoch different with current actual epoch
        # Will be updated on next_train()
        self.shared_train_epoch  = multiprocessing.Value('i', -1)
        self.process_train_epoch = -1
        self.process_train_index = -1
        self.process_val_index   = -1

    def build(self, **kwargs):
        self.train_path     = os.path.join(self.dataset_path, 'train')
        self.val_path       = os.path.join(self.dataset_path, 'val')
        self.align_path     = os.path.join(self.dataset_path, 'align')
        self.build_dataset()
        # Set steps to dataset size if not set
        self.steps_per_epoch  = self.default_training_steps if self.steps_per_epoch is None else self.steps_per_epoch
        self.validation_steps = self.default_validation_steps if self.validation_steps is None else self.validation_steps
        return self

    @property
    def training_size(self):
        return len(self.train_list)

    @property
    def default_training_steps(self):
        return self.training_size / self.minibatch_size

    @property
    def validation_size(self):
        return len(self.val_list)

    @property
    def default_validation_steps(self):
        return self.validation_size / self.minibatch_size

    def enumerate_videos(self, path):
        video_list = []
        for video_path in glob.glob(path):
            video_list.append(video_path)
        return video_list

    def enumerate_align_hash(self, video_list):
        align_hash = {}
        for video_path in video_list:
            video_id = os.path.splitext(video_path)[0].split('/')[-1]
            align_path = os.path.join(self.align_path, video_id)+".align"
            align_hash[video_id] = Align(self.absolute_max_string_len, text_to_labels).from_file(align_path)
        return align_hash

    def build_dataset(self):
        print "\nEnumerating dataset list from disk..."
        self.train_list = self.enumerate_videos(os.path.join(self.train_path, '*', '*'))
        self.val_list   = self.enumerate_videos(os.path.join(self.val_path, '*', '*'))
        self.align_hash = self.enumerate_align_hash(self.train_list + self.val_list)
	print self.align_hash
        print "Found {} videos for training.".format(self.training_size)
        print "Found {} videos for validation.".format(self.validation_size)
        np.random.shuffle(self.train_list)

    def get_align(self, _id):
        return self.align_hash[_id]
    
    def get_output_size(self):
        return 28

    def get_batch(self, index, size, train):
        if train:
            video_list = self.train_list
        else:
            video_list = self.val_list
	def get_list_safe(l, index, size):
            ret = l[index:index+size]
            while size - len(ret) > 0:
                ret += l[0:size - len(ret)]
            return ret

        X_data_path = get_list_safe(video_list, index, size)
	#print "X_data_path",X_data_path
        X_data = []
        Y_data = []
        label_length = []
        input_length = []
        source_str = []
        for path in X_data_path:
	    #print "path",path
            video = Video().from_frames(path)
            align = self.get_align(path.split('/')[-1])
            video_unpadded_length = video.length
	    #print "vavl",video,align,video_unpadded_length
	    #print "vavlnext",video.data,align.padded_label,align.label_length,align.sentence
            X_data.append(video.data)
            Y_data.append(align.padded_label)
            label_length.append(align.label_length) 
            input_length.append(video.length) 
            source_str.append(align.sentence) 
        source_str = np.array(source_str)
        label_length = np.array(label_length)
        input_length = np.array(input_length)
        Y_data = np.array(Y_data)
        X_data = np.array(X_data).astype(np.float32) / 255 # Normalize image data to [0,1],

        inputs = {'the_input': X_data,
                  'the_labels': Y_data,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function

        return (inputs, outputs)

    @threadsafe_generator
    def next_train(self):
        r = np.random.RandomState(self.random_seed)
        while 1:
            with self.cur_train_index.get_lock(), self.shared_train_epoch.get_lock():
                cur_train_index = self.cur_train_index.value
		'''print "cur_t_in",cur_train_index
		print "curtraindval",self.cur_train_index.value
		print "minib",self.minibatch_size'''
                self.cur_train_index.value += self.minibatch_size
		#print "curtraindval",self.cur_train_index.value,cur_train_index
                if cur_train_index >= self.steps_per_epoch * self.minibatch_size:
                    cur_train_index = 0
                    self.shared_train_epoch.value += 1
                    self.cur_train_index.value = self.minibatch_size
		    #print "one"
                if self.shared_train_epoch.value < 0:
                    self.shared_train_epoch.value += 1
		    #print "two"
                if self.cur_train_index.value >= self.training_size:
                    self.cur_train_index.value = self.cur_train_index.value % self.minibatch_size
		    #print "three"
                epoch_differences = self.shared_train_epoch.value - self.process_train_epoch
            if epoch_differences > 0:
                self.process_train_epoch += epoch_differences
                for i in range(epoch_differences):
                    r.shuffle(self.train_list)

            '''if self.curriculum is not None and self.curriculum.epoch != self.process_train_epoch:
                self.update_curriculum(self.process_train_epoch, train=True)'''

            ret = self.get_batch(cur_train_index, self.minibatch_size, train=True)

            yield ret

    @threadsafe_generator
    def next_val(self):
        while 1:
            with self.cur_val_index.get_lock():
                cur_val_index = self.cur_val_index.value
                self.cur_val_index.value += self.minibatch_size
                if self.cur_val_index.value >= self.validation_size:
                    self.cur_val_index.value = self.cur_val_index.value % self.minibatch_size
            '''if self.curriculum is not None and self.curriculum.epoch != self.process_epoch:
                self.update_curriculum(self.process_epoch, train=False)'''

            ret = self.get_batch(cur_val_index, self.minibatch_size, train=False)
            yield ret

    def on_train_begin(self, logs={}):
        with self.cur_train_index.get_lock():
            self.cur_train_index.value = 0
        with self.cur_val_index.get_lock():
            self.cur_val_index.value = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.process_epoch = epoch
