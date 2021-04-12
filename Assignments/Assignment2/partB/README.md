### Approach

- We used various models pre-trained on imagenet dataset like InceptionResNetV2, InceptionV3, Xception and ResNet.
- We first removed all the fully connected layers from each of this layers, froze the base, trained on iNaturalist Dataset and then fine-tuned it.
- We were able to acheive 82% accuracy on validation dataset (10%) and around 78-80% on Test Dataset.
- For the initial part, we scaled the images down to (256,256,3) and normalized each pixel in [-1,1].
- Also, we used a global average pooling layer to reduce the params needed for the fully connected layer.

### Plots

- Plots can be found on https://wandb.ai/rudradesai200/Assignment-partB/reports/Assignment2--Vmlldzo1OTY3NjQ#solution

### Inferences

- Transfer learning on pretrained model performs a lot better than training models from scratch, as the pre-trained models are trained to identify the proper characteristics of the image. All the pre-trained models we used were trained to identify 1000 different classed present in ImageNet Dataset. So, all the CNN filter used in these models learnt to properly differentiate between various types of object. This domain knowledge helps a lot in our case of iNaturalist dataset, which is also an image classification task.
- Seeing the charts above, we can confirm that the approach 2 - Freeze, train and fine-tune works a lot better than the approach 1.
- The main reason is that fine-tuning helps to learn the model properly over the new dataset. Freezing the bottom layers gives us a good base for pre-raining, but without fine-tuning, the model won't learn the new dataset properly.
- Also, the order of learning was InceptionResNetV2 > InceptionV3 > Xception > ResNet50.
- InceptionResNetV2 works better than InceptionV3 because of the added residual connections (skip connections). These connections allowed the network to become more deeper and also learn the shallow connections well.
- Xception is a modularized and smaller version of Inception, so it was expected to perform worse as compared to the Inception model.
- Inception class models uses the idea of using multiple different types of filter like Maxpool, 3x3, 5x5, etc in the same layer by first using 1x1 filter and then stitching them all together. This idea revolutionalized CNN models as now there is no need to use larger filters , instead one can use multiple smaller filters to achieve a better result. This was the reason why Inception class models performed better than ResNet.
- It can also be seen that higher the number of epochs, the model tends to over-fit the dataset because of the huge capacity these model contains. So, number of epochs between 3-5 works well for training and 1-2 for fine-tuning.
- Moreover, the sudden change in losses in the above graphs is due to the fine-tuning step in experiment 2.
