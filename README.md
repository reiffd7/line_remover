# Line Remover - Applied to Uncle Peter's Sailboats
A project using CNNs to remove ruled lines from sketches


__          |  __ | __
:-------------------------:|:-------------------------: | :-------------------------:
![uncle_peter](/presentation_images/uncle_peter.png)  |  ![uncle_peter](/presentation_images/uncle_peter1.png)|![uncle_peter](/presentation_images/uncle_peter2.png)

1. [Overview](#overview)
2. [The Strategy](#the-strategy)
2. [Data](#data)
3. [EDA](#eda)
4. [data-classification](#data-classification)
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

Due to the difference in quality across the images, I could find a process by which to augment lines in a way that would be similar to ruled images. I would also have to resize these very large images, degrading the quality




## EDA

## Data Classification

## CNN Structure

## CNN Results

## Picture Scrubbing

## Results

## Further Work

## Sources




