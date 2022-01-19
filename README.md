# ISIC-Skin-Cancer-Classification-Using-Deep-Learning

Uses ISIC Dataset - RGB images and imbalanced dataset and loading data from directory

The goal for ISIC 2019 is classify dermoscopic images among nine different diagnostic categories:

1. Melanoma
2. Melanocytic nevus
3. Basal cell carcinoma
4. Actinic keratosis
5. Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)
6. Dermatofibroma
7. Vascular lesion
8. Squamous cell carcinoma
9. None of the others

25,331 images are available for training across 8 different categories. Additionally, the test dataset (planned release August 2nd) will contain an additional outlier class not represented in the training data, which developed systems must be able to identify. 

But for the sake of training in low resource system, only 6000 images were considered.

basicModel uses a newly created model and achieves an accuracy of near about 70%.
preTrainedModel used a pre-trained Xception model along with a few extra layers added in the end and achieves and accuracy of near about 74%.

Dataset is imbalanced and only nearly a smaller dataset of 6007 images have been used instead of the original dataset with 25331 images due to lesser computation power of CPU.
