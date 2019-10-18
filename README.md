# Line Remover - Applied to Uncle Peter's Sailboats
A project using CNNs to remove ruled lines from sketches


__          |  __ | __
:-------------------------:|:-------------------------: | :-------------------------:
![uncle_peter](/presentation_images/uncle_peter.png)  |  ![uncle_peter](/presentation_images/uncle_peter1.png)|![uncle_peter](/presentation_images/uncle_peter2.png)

1. [Overview](#overview)
2. [The Strategy](#the-strategy)
2. [Data](#data)
3. [EDA](#eda)
4. [Data Classification](#data-classification)
5. [CNN Structure](#cnn-structure)
5. [CNN Results](#cnn-results)
6. [Picture Scrubbing](#picture-scrubbing)
7. [Results](#results)
8. [Further Work](#further-work)
10. [Sources](#sources)

![classification_gif](/notebooks/savefig/gif_of_images.gif)


## Overview
Thank you to Land Belenky for the project idea. Land's Uncle Peter accumulated 733 sailboat pencil sketches over the years. Unfortunately, 513 of these images were done on ruled paper. Can we salvage the ruled pictures?

Hypothesis: It is possible to train a CNN to remove ruled lines from an image without apparent degradation of the image


Goals

<li> Classify whether or not the central pixel of an individual frame of a picture belongs to a drawing or a line </li>
<li> Train a CNN to predict whether or not a pixel belongs to a line</li>
<li> Use the trained CNN to scrub a picture of its lines</li>
<li> Maintain the quality of the image throughout the process </li>


## The Strategy


At first, I considered augmenting lines onto unruled images. I would then train an autoencoder to remove lines from these images. Finally, I would apply the model to ruled images.

![uncle_peter](/presentation_images/first_look.png)

Due to the difference in quality across the images, I could find a process by which to augment lines in a way that would be similar to ruled images. I would also have to resize these very large images, degrading the quality.

I found a new strategy from a paper that showed how it was possible to remove staff lines from music scores. Instead of looking at the entire picture, this paper looked at 28 x 28 windows, classified if the central pixel came from a staff or symbol. A CNN was then trained on staff and symbol classifications. Staff classification were then removed. The results were impressive. 

  Windows | Results
:-------------------------:|:-------------------------: 
![windows](/presentation_images/windows.png)  |  ![staff_removal](/presentation_images/staff_removal.png)

## EDA

To execute my strategy, I wanted to start by binarizing each image. This way, I could more clearly classify pixels. I started by converting all images to greyscale. Greyscale standardized each image and did not degrade the quality of each image due to the monochromatic nature of the images. 

Binarization across a large span of images was only possible after further standardization. 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;x_i=\frac{x_i - x_{min}}{x_{max} - x_{min}}" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />

My standardization technique reduced each pixel to a value between 0 and 1 which made for easy visualization. 



Greyscale | Binarized
:-------------------------:|:-------------------------: 
![greyscale](/presentation_images/greyscale.png)  |  ![staff_removal](/presentation_images/binarized.png)

![standardized](/presentation_images/eda.png)

![lines](/presentation_images/eda2.png)

On a pixel-by-pixel basis lines and drawings appear to be distinguished


## Data Classification

![giphy](/presentation_images/gif_of_my_images.gif)

I wanted to create as much data as possible to train my model, so I attempted to automize the data classification process.

I noticed that a frame displaying a line has a clear signature on a sobel filter. The magnitude of mean sobel values above the line is roughly equal to the magnitude of mean sobel values below the line. This made for my first classification criteria. Is there a tight gradient of sobel values?

I also noticed that a frame displaying a line is roughly less than 30% black pixels. This made for my second classfication criteria. 

As I iterated through pixels, first I would look for a tight gradient. If it exists, I would declare the frame a line. If it does not exist, I would consider the pixel %. If that was less than 30%, I would declare the frame a line. 


## CNN Structure
![giphy](/presentation_images/cnn.png)

I designed my CNN through a process of trial and error. Throughout the process, I observed that increasing number of filters from 32 to 64 increased accuracy. For the most part, my model was very accurate (>95% accuracy on my validation set) from the start. However, my precision and recall rates were at 0. I realized, this was because I had set my batch size to 1. 


## CNN Results

## Picture Scrubbing

## Results

## Further Work

## Sources




