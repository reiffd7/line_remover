import numpy as np 
from scipy.misc import imread
import glob
from PIL import Image, ImageOps

# '''
# These functions are used to remove borders from image lists and individual images
# '''

def shear(image_list):
    for image in image_list:
        im = Image.open(image)
        im = ImageOps.crop(im, 10)
        im.save(image)

def shear_single(filepath):
    im = Image.open(filepath)
    im = ImageOps.crop(im, 10)
    im.save(filepath)



if __name__ == '__main__':
    line_images = glob.glob('../lines/*')
    drawing_images = glob.glob('../drawings/*')
    shear(line_images)
    shear(drawing_images)
    
        # for image in line_images:
        #     im = Image.open(original_file_path)