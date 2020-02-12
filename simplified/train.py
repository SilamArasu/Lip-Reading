from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from generators import BasicGenerator
#from callbacks import Statistics, Visualize
#from curriculums import Curriculum
#from decoders import Decoder
#from helpers import labels_to_text
#from spell import Spell
from model2 import LipNet
import numpy as np
import datetime
import os
import sys

np.random.seed(55)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

def train(run_name, speaker, start_epoch, stop_epoch, img_c, img_w, img_h, frames_n, absolute_max_string_len, minibatch_size):
    DATASET_DIR = os.path.join(CURRENT_PATH, speaker, 'datasets')
    OUTPUT_DIR = os.path.join(CURRENT_PATH, speaker, 'results')

    #curriculum = Curriculum(curriculum_rules)
    lip_gen = BasicGenerator(dataset_path=DATASET_DIR,
                                minibatch_size=minibatch_size,
                                img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                                absolute_max_string_len=absolute_max_string_len, start_epoch=start_epoch).build()

    lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                            absolute_max_string_len=absolute_max_string_len, output_size=lip_gen.get_output_size())

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    
    if not os.path.exists(os.path.join(OUTPUT_DIR, run_name)):
    	os.makedirs(os.path.join(OUTPUT_DIR, run_name))
    checkpoint  = ModelCheckpoint(os.path.join(OUTPUT_DIR, run_name, "weights{epoch:02d}.h5"), monitor='val_loss', save_weights_only=True, mode='auto', period=1)

    lipnet.model.fit_generator(generator=lip_gen.next_train(),
                        steps_per_epoch=lip_gen.default_training_steps, epochs=stop_epoch,
                        validation_data=lip_gen.next_val(), validation_steps=2,
callbacks=[checkpoint, lip_gen],
                        initial_epoch=start_epoch,
                        verbose=1,
                        max_q_size=5,
                        workers=2,
                        pickle_safe=True)
    
if __name__ == '__main__':
    run_name = "model"
    speaker = sys.argv[1]
    train(run_name, speaker, 0, 1, 3, 100, 50, 75, 32, 2)
