# AutoML Vision

For this multinomial image classification competition, [Google Cloud AutoML Vision](https://cloud.google.com/vision/) was used to rapidly produce an image recognition model.

## Competition Overview

The [Plant Seedlings Classification](https://www.kaggle.com/c/plant-seedlings-classification) competition on Kaggle requires 794 test images of plant seedlings to be labelled with one of twelve somewhat-balanced types of plant species. The predictions are evaluated using a micro-averaged F1 score. The distribution of the 4780 training images amongst these classes is shown in the screenshot below:

![Distribution of plant species in training data](screenshots/1_label_stats.PNG#center?raw=true "Training data distribution")

## Model Training

The training images were uploaded to the AutoML Vision web API interface, and the model was then training using one hour of free compute resource. A more accurate model could probably be built by allowing training to occur for up to 24 hours (although at the time of writing, training beyond one hour is charged at USD$20 per hour).

## Results on Training Data

Precision and recall of 91.6% and 90.2% respectively were achieved by the model on the training data using cross validation. Further detail is provided in the screenshots below:


![High level cross validation results](screenshots/2_high_level_results.PNG#center?raw=true "High level cross validation results")


![Precision Recall Curve](screenshots/3_all_labels.PNG#center?raw=true "Precision Recall Curve")

![Confusion Matrix](screenshots/4_confusion_matrix.PNG#center?raw=true "Confusion Matrix")

## Predictions for Test Data

[This notebook](AutoML_predictions.ipynb) shows a simple script that was used to obtain predictions for the 794 test images. For some test images, the model failed to give a prediction that met the 0.50 threshold - in these cases, a default label of 'loose_silky_bent' (the most populous class in the training data) was assigned. It took approximately 20 minutes to generate all of the predictions.

## Results on Test Data

A micro-averaged F1 score of 86% was achieved when the test data predictions were submitted to the Kaggle competition. This would place in the 75th percentile of the public leaderboard. Although this is far from coming close to the best performing models, considering this model required approximately one hour of developer time and 1.5 hours of compute time to run from start to finish, this result is quite impressive.

![Score of AutoML Vision model on the Kaggle test dataset](screenshots/5_kaggle_score.PNG?raw=true "Kaggle Score")
