# SIGIR 2020 E-Commerce Workshop Data Challenge (Multi-modal Classification Task) 
## Task Intro

This challenge focuses on the topic of large-scale multi-modal (text and image) classification, where the goal is to predict each product’s type code as defined in the catalog of Rakuten France. To be specific, given a training set of products and their product type codes, predict the corresponding product type codes for an unseen held out test set of products. The systems are free to use the available textual titles and/or descriptions whenever available and additionally the images to allow for true multi-modal learning.

## Dataset 
The organizer released approximately 99K product listings in tsv format, including 84,916 samples for training, 937 samples for phase 1 testing and 8435 samples for phase 2 testing. The training dataset consists of product titles, descriptions, images and their corresponding product type codes. There are 27 product categories in the training dataset and the number of product samples in each category ranges from 764 to 10,209. The frequency distribution of categories in the training dataset is shown in the following figure.
![category frequency distribution](assets/data_dist.png)


## Preprocess
We divide the full 84916 labeled samples randomly into training and validation set at a ratio of 9:1. For different modal data, we perform some basic preprocessing techniques as follows:
- **Text preprocessing**: We simply remove some HTML tags like `L&#39` and `<p>` from product title and description texts.  
- **Image preprocessing**: The images are preprocessed during the training process of image classifier. The preprocessing techniques includes resize, random crop, rotation and horizontal flip.