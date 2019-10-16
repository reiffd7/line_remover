import numpy as np
import matplotlib.pyplot as plt
import glob


from skimage import io, color, filters, feature, restoration
from skimage.transform import resize, rotate
from skimage.color import rgb2gray
from scipy.spatial.distance import squareform
from transparent_imshow import transp_imshow
import matplotlib.cm as cm
from scipy import ndimage, misc
from src.image_shear import shear_single
from src.standardizer import standardizer


class imageGenerator(object):

    def __init__(self, image):
        self.image = image

    def pad(self, rows, cols):
        result = np.ones((rows,cols))
        result[:self.image.shape[0], :self.image.shape[1]] = self.image
        self.image = result

    def zoom(self, row, col):
        self.zoom = self.image[row:row+1000, col:col+1000]


    def classify(self, row, start, end, prefix):
        for i in range(start, end):
            size = 30
            row_index = row
            col_index = i
            masked_window = np.random.random((self.zoom.shape[0],self.zoom.shape[1]))
            masked_window[row_index:row_index+size, col_index:col_index+size] = 1
            masked_window = np.ma.masked_where(masked_window != 1, masked_window)

            masked_pixel = np.random.random((self.zoom[row_index:row_index+size, col_index:col_index+size].shape[0], self.zoom[row_index:row_index+size, col_index:col_index+size].shape[1]))
            masked_pixel[15,15] = 1
            masked_pixel = np.ma.masked_where(masked_pixel != 1, masked_pixel)

            masked_pixel1 = np.random.random((self.zoom[row_index:row_index+size, col_index:col_index+size].shape[0], self.zoom[row_index:row_index+size, col_index:col_index+size].shape[1]))
            masked_pixel1[11:19, 11:19] = 1
            masked_pixel1 = np.ma.masked_where(masked_pixel1 != 1, masked_pixel1)

            window = self.zoom[row_index:row_index+size, col_index:col_index+size]
            colored_percentage = np.count_nonzero(window==0)/(30**2)
            pixel_value = window[15, 15]
            window_sobel = ndimage.sobel(window, axis=0)
            above_area = np.mean(window_sobel[10:15, 11:19])
            below_area = np.mean(window_sobel[15:20, 11:19])
            sobel_value = window_sobel[15, 15]
            # Overlay the two images
            fig, ax = plt.subplots(1, 3)
            ax.ravel()
            ax[0].imshow(self.zoom, cmap='gray')
            ax[0].imshow(masked_window, cmap='prism', interpolation='none')
            # ax[0].imshow(masked_pixel, cmap=cm.jet, interpolation='none')
            ax[1].imshow(window, cmap='gray')
            # ax[1].imshow(masked_pixel, cmap='prism', interpolation='none')
            ax[1].set_title('Pixel Value: {}'.format(pixel_value))
            ax[2].imshow(window_sobel)
            ax[2].imshow(masked_pixel1, cmap='jet')
            ax[2].imshow(masked_pixel, cmap='prism', interpolation='none')
            extent = ax[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            l_filepath = f'../data/train/lines/{prefix}{i:04}'
            d_filepath = f'../data/train/drawings/{prefix}{i:04}'

            if pixel_value == 0.0:
                print(above_area)
                print(below_area)
                print(above_area - -below_area)
                if np.abs(above_area) >= 0.7 and np.abs(below_area) >= 0.7:
                    if np.abs(above_area - -below_area) < 0.35:
                        ax[2].set_title('Line')
                        fig.savefig(l_filepath, bbox_inches=extent)
                        shear_single(l_filepath)
                    else:
                        if colored_percentage < 0.3:
                            ax[2].set_title('Line')
                            fig.savefig(l_filepath, bbox_inches=extent)
                            shear_single(l_filepath)
                        else:
                            ax[2].set_title('Drawing')
                            fig.savefig(d_filepath, bbox_inches=extent)
                            shear_single(d_filepath)
                elif np.abs(above_area) >= 0.1 and np.abs(below_area) >= 0.1: 
                    if np.abs(above_area - -below_area) < 0.01:
                        ax[2].set_title('Line')
                        fig.savefig(l_filepath, bbox_inches=extent)
                        shear_single(l_filepath)
                    else:
                        if colored_percentage < 0.3:
                            ax[2].set_title('Line')
                            fig.savefig(l_filepath, bbox_inches=extent)
                            shear_single(l_filepath)
                        else:
                            ax[2].set_title('Drawing')
                            fig.savefig(d_filepath, bbox_inches=extent)
                            shear_single(d_filepath)
                else:
                    if colored_percentage < 0.3:
                            ax[2].set_title('Line')
                            fig.savefig(l_filepath, bbox_inches=extent)
                            shear_single(l_filepath)
                    else:
                        ax[2].set_title('Drawing')
                        fig.savefig(d_filepath, bbox_inches=extent)
                        shear_single(d_filepath)
            print(i)

    








if __name__ == '__main__':
    ruled = glob.glob('../Sketches/Ruled/*')

    images_30 = standardizer(ruled)
    images_30.greyscale(30)
    images_30.standardize()
    images_30.binarize(0.75)
    binarized0 = images_30.binarized_images[0]

    ig_0 = imageGenerator(binarized0)
    ig_0.pad(4017, 3600)
    ig_0.zoom(1500, 1500)
    ig_0.classify(103, 900)
    # result = np.ones((4017,3600))
    # result[:binarized0.shape[0], :binarized0.shape[1]] = binarized0
    # zoom = result[1500:2500, 1500:2500]
    # io.imshow(ig_0.zoom)
    # plt.show()    