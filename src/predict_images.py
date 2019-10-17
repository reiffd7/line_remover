from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from CNN import imageCNN
from standardizer import standardizer
from image_shear import shear_single
import glob
import os
import sys
from skimage import io, color, filters, feature, restoration
this_file = os.path.realpath(__file__)
SCRIPT_DIRECTORY = os.path.split(this_file)[0]
ROOT_DIRECTORY = os.path.split(SCRIPT_DIRECTORY)[0]
MODEL_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'models')  
FRAMES_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data/to_predict')
sys.path.append(ROOT_DIRECTORY)


def zoom(row, col, image):
        return image[row:row+1000, col:col+2000]

class lineScrubber(object):

    def __init__(self, image, model_path):
        self.image = image
        self.image_rows = image.shape[0]
        self.image_cols = image.shape[1]
        self.model = load_model(model_path)


    def _create_and_save_fig(self, window, filepath):
        fig, ax = plt.subplots(1, 1)
        ax.imshow(window, cmap='gray')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filepath, bbox_inches=extent)
        shear_single(filepath)

    def _predict_fig(self, filepath, target_size=(30, 30)):
        img = image.load_img(filepath, target_size=(target_size[0], target_size[1]))
        x = image.img_to_array(img)
        x = x.reshape(-1, 30, 30, 3)
        return self.model.predict(x)[0][0]

    def _alter_figure(self, i, j, prediction):
        if prediction == 1.0:
            self.image[i+15, i+15] = 1.0
            print('pixel changed')


    def scrubber(self, size=30):
        for i in range(self.image_rows-(size+1)):
            print('Were on row: {}'.format(i))
            for j in range(self.image_cols-(size+1)):
                window = self.image[i:i+size, j:j+size]
                pixel_val = window[15, 15]
                print('Pixel Value: {}'.format(pixel_val))
                if pixel_val = 0.0
                    print('Black pixel!')
                    filepath = os.path.join(FRAMES_DIRECTORY, '{}pix.png'.format(i*j))
                    self._create_and_save_fig(window, filepath)
                    print('created fig')
                    prediction = self._predict_fig(filepath)
                    print('modeled fig')
                    self.alter_figure(i, j, prediction)


    





if __name__ == '__main__':
    ruled = glob.glob(os.path.join(ROOT_DIRECTORY, 'Sketches/Ruled/*'))
    images = standardizer(ruled)
    images.greyscale(6)
    images.standardize()
    print('Standardized')
    images.binarize(0.7)
    print('Binarized')
    
    
    image = images.binarized_images[4]
    zoom = zoom(1300, 100, image)