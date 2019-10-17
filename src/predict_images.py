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
RESULTS_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data/scrubbed_images')
sys.path.append(ROOT_DIRECTORY)


def zoom(row, col, image):
        return image[row:row+1000, col:col+2000]

class lineScrubber(object):
    '''
    This class takes in a binarized image, a cnn, model, and a figure name. 
    It iterates through every pixel of the image and predicts if it is a line or
    a drawing. If the image is a line, the pixel is removed. After this process, the 
    scrubbed figure is saved. 
    '''

    def __init__(self, fig, model_path, figname):
        self.fig = fig
        self.fig_rows = fig.shape[0]
        self.fig_cols = fig.shape[1]
        self.model = load_model(model_path)
        self.figname = figname


    def _create_and_save_fig(self, window, filepath):
        fig, ax = plt.subplots(1, 1)
        ax.imshow(window, cmap='gray')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filepath, bbox_inches=extent)
        border = (70, 5, 70, 5)
        shear_single(filepath, border)

    def _predict_fig(self, filepath, target_size=(30, 30)):
        img = image.load_img(filepath, target_size=(target_size[0], target_size[1]))
        x = image.img_to_array(img)
        x = x.reshape(-1, 30, 30, 3)
        return self.model.predict(x)[0][0]

    def _alter_figure(self, i, j, prediction):
        print(prediction)
        if prediction == 1.0:
            self.fig[i+15, j+15] = 1.0
            print('pixel changed')

    def save_fig(self, filepath_csv, filepath_img):
        np.savetxt(filepath_csv, self.fig, delimiter=',')
        fig, ax = plt.subplots(1, 1)
        ax.imshow(self.fig, cmap='gray')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filepath_img, bbox_inches=extent)
    


    def scrubber(self, size=30):
        self.save_fig(os.path.join(RESULTS_DIRECTORY, '{}before.csv'.format(self.figname)), os.path.join(RESULTS_DIRECTORY, '{}before.png'.format(self.figname)))
        for i in range(self.fig_rows-(size+1)):
            print('Were on row: {}'.format(i))
            for j in range(self.fig_cols-(size+1)):
                window = self.fig[i:i+size, j:j+size]
                pixel_val = window[15, 15]
                print('Pixel Value: {}'.format(pixel_val))
                if pixel_val == 0.0:
                    print('Black pixel!')
                    filepath = os.path.join(FRAMES_DIRECTORY, '{}pix.png'.format(i*j))
                    self._create_and_save_fig(window, filepath)
                    print('created fig')
                    prediction = self._predict_fig(filepath)
                    print('modeled fig')
                    self._alter_figure(i, j, prediction)
                print("({}, {})".format(i, j))
        self.save_fig(os.path.join(RESULTS_DIRECTORY, '{}result.csv'.format(self.figname)), os.path.join(RESULTS_DIRECTORY, '{}result.png'.format(self.figname)))

   


    





if __name__ == '__main__':
    ruled = os.path.join(ROOT_DIRECTORY, 'Sketches/Ruled/*')
    image_path = os.path.join(ROOT_DIRECTORY, 'Sketches/Ruled/Sketch_Page_108.jpg')
    images = standardizer(ruled, image_path)
    # images.greyscale(6)
    images.greyscale_one()
    images.standardize_one()
    # images.standardize()
    print('Standardized')
    images.binarize_one(0.7)
    print('Binarized')
    # images.visualize(1, 5)
    
    model_path = os.path.join(MODEL_DIRECTORY, 'model_names/test5.h5')
    
    figure = images.binarized_image
    zoom = zoom(1300, 100, figure)


    image_scrubber = lineScrubber(zoom, model_path, 'test')
    image_scrubber.scrubber()

    # row = 348
    # col = 359
    # window = zoom[1: 31, 1:31]
    # fig, ax = plt.subplots(1, 1)
    # ax.imshow(zoom, cmap='gray')
    # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # filepath = os.path.join(RESULTS_DIRECTORY, 'test2.png')
    # fig.savefig(filepath, bbox_inches=extent)
    # border = (70, 5, 70, 5)
    # shear_single(filepath, border)