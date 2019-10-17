from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage import io, color, filters, feature, restoration
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
this_file = os.path.realpath(__file__)
SCRIPT_DIRECTORY = os.path.split(this_file)[0]
ROOT_DIRECTORY = os.path.split(SCRIPT_DIRECTORY)[0]
sys.path.append(ROOT_DIRECTORY)


class imageGenerator(object):
    '''
    This class takes a path to a directory of images, the directory we want to
    export images to, and the number of images we want to create. With that 
    information, we generate new images that are our original images but skewed,
    zoomed, rotated, etc. The new images are saved into our export directory.
    '''
    

    def __init__(self, import_path, export_path, n):
        self.import_path = import_path
        self.export_path = export_path
        self.n = n
        self.create_and_save_images()

    def create_img_generator(self):
        # ''' this creates the image generator object'''
        self.datagen = ImageDataGenerator(
                        rotation_range=5,
                        width_shift_range=0.2,
                        height_shift_range=0.05,
                        rescale=1./255,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode='nearest')
        print('generator created')


    def create_pictures(self):
        # '''This will run through the generator created
        # in the create_img_generator function to actually save the images'''

        i = 0
        for batch in self.datagen.flow_from_directory(
                                        directory=self.import_path,
                                        save_to_dir=self.export_path,
                                        save_prefix='keras_',
                                        save_format='png',
                                        batch_size=1):
            i += 1
            if i == self.n:
                break

    def create_and_save_images(self):
        self.create_img_generator()
        self.create_pictures()




if __name__ == '__main__':
    print('Hey')
    imageGenerator(os.path.join(ROOT_DIRECTORY, 'data/train'), 
                    os.path.join(ROOT_DIRECTORY, 'data/generated'), 100)

    # imageGenerator(os.path.join(ROOT_DIRECTORY, 'drawings'), 
    #                 os.path.join(ROOT_DIRECTORY, 'generated_drawings'), 100)


    