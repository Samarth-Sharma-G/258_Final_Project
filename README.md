[Project Report Link](https://docs.google.com/document/d/11hbjZfpXR2C6PgKxpLrhuE5oDf72uBq2FPDUJGOtfrU/edit?usp=sharing)

[Presentation Slide deck](https://docs.google.com/presentation/d/17taN_pZp_4o1ZOrtxuZ7G69Y7ex3MhhvP2R-f0azNMU/edit?usp=sharing)

# Detection of Defects in Computer Chips and MLOps

A wafer chip, or semiconductor wafer, is a thin slice of material like silicon used to create integrated circuits and microdevices through processes such as photolithography and doping. Defective wafer chips can lead to degraded product performance, reliability, and safety, and have significant economic, reputational, environmental, and supply chain impacts. Therefore, detecting and addressing defects is essential for maintaining high standards in semiconductor manufacturing and ensuring the quality and reliability of electronic devices.

## EDA and Modifications :

<a href='https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map'> Link to Dataset </a>

Data has 811,457 wafer maps collected from 46,393 lots in real-world fabrication. This extrapolated to 30078 images of size 26*26.

Sample of 3 wafer chips side by side:

![image (3)](https://github.com/Samarth-Sharma-G/258_Final_Project/assets/107587243/a3ed05e3-dbed-4fb0-9f4d-fba32531e66c)

All the Classes: Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full, none, []. 

Distribution: 

[]: 15712 (Unlabeled data, we simply drop it)

None: 13489 (Images labeled no defect)

All Defects Combined: 877 (Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full)

![image](https://github.com/Samarth-Sharma-G/258_Final_Project/assets/107587243/c9bbb807-d180-4f89-97ef-8c01193feb45)

As we had a major class imbalance 
- we decided to downsample the None class to 877 random images 
- we combined all the defects into one Defective Class which also accounted to 877 images establishing balanced classes

Thus, moving to a binary classification problem.

## Modeling

Evaluation Metric: Accuracy (As we converted the problem to a binary classification problem with balanced classes, for simplicity sake we've choosen accuracy as our evaluation metric.)

Scenarios in which Precision/Recall would be preferred:
- Recall: In case it's fine to falsly flag wafer chips and there is another layer of check where the good ones wil be retrived (or it's fine to compromise some good ones for quality assurance).
- Precision: In case the chips are really expensive and this is the last layer of check, so we can't afford to falsly flag a chip defective and just discard it (less likely).


1. For our initial model, we have the basic architecture with an input layer, 3 conv layers, flatten layer, 2 fully connected layers and output layer and the optimizer Adam.   and after training this model, the loss curve we had was

![image](https://github.com/Samarth-Sharma-G/258_Final_Project/assets/107587243/b924007f-9d94-44b9-87cb-c3914273fa1f)

In this plot, we observe the training and validation loss of our initial model over 10 epochs. The training loss (blue line) decreases steadily, indicating that the model is effectively learning from the training data. The validation loss (orange line) follows a similar trend initially but then diverges after a few epochs. This divergence indicates the overfitting, where the model performs well on the training data but not on the validation data.

Observation:
- The model is overfitting the training data

Improvements to be incorporated:
- Regularization
- Improving model Architecture


Then we thought of adding regularization techniques such as Dropout layers and implementing early stopping to prevent overfitting. At the same time this initial architecture doesn't follow the standard architetcure where a Conv Layer is followed by pooling layer. And then we started experimenting by making changes to the architecture by adding pooling layers and dropout layers.


2. After adding these, we observed an improvement where the initial training was slower, but eventually showed better generalization with reduced overfitting compared to the initial model. and also the loss and accuracy showed more consistent improvement.

3. Then we tried changing the optimizer from Adam to Adadelta with a learning rate of 0.01. It struggled with accuracy, and both training and validation loss plateaued early.



Best Model's Performance after incorporating the changes:

4. Changed the optimizer to AdamW and it has pooling layers, dropout layers and it Demonstrated significant improvement in both training and validation metrics and also Better generalization with consistent improvement in val_loss and val_accuracy.


Specific Changes:
- Using different regularization techniques such as Batch Normalization and Dropout layer
- Reducing the model complexity - specially after flattening the input
- Using newer more adaptive optimizer AdamW

![image](https://github.com/Samarth-Sharma-G/258_Final_Project/assets/107587243/aa427390-3989-4a64-8b10-4e281b9afc67)



In this plot, we observe the training and validation loss of Model 4 over 10 epochs. The training loss (blue line) shows a steady decrease, indicating that the model is effectively learning from the training data. The validation loss (orange line) also decreases significantly, particularly around epoch 3, suggesting that the model is generalizing well to unseen data. The continued decrease of the validation loss towards the later epochs indicates that the model is not overfitting and is performing well on both the training and validation sets. The final low validation loss highlights the model's strong predictive capability for detecting defects in chip images.

Observation:

Our objective for model development phase was to create a model with 90+ performance (accuracy here), so we stop even though if we trained the model for more epoch the performance would have improved as the loss is still going down. 

Baseline vs Best Model Architectures:

![Baseline Model Architecture (1)](https://github.com/Samarth-Sharma-G/258_Final_Project/assets/107587243/1163b71e-ded1-45ac-bc7d-1f70fbd0976d)

## MLOps

For the MLops, we're assuming that the data cleaing pipline is different, which will channel data to the data store. Thus, alwalys only the clean and updated data will be there in the data store.

![image](https://github.com/Samarth-Sharma-G/258_Final_Project/assets/107587243/2cc6b1e7-88f6-4b48-a4bf-67caff2fad3c)

This pipeline can be scheduled to run on regular intervals , or run when new data arrives. It can configure directly as a pipeline on Azure ML.

An experiment is nothing but a collection of different models you built when trying to find the best one:

![image](https://github.com/Samarth-Sharma-G/258_Final_Project/assets/107587243/3bd47fc6-a073-45bc-9a11-528947871c53)

With each experiment, we have the list of all the models trained for it, we use MLFlow for logging models performance and its artifacts

![image](https://github.com/Samarth-Sharma-G/258_Final_Project/assets/107587243/553fdd6b-b746-4ca6-be64-5e746639569d)

For each logged model we have all the logged artifacts and metrics, along with the hyperparameter used for training it

![image](https://github.com/Samarth-Sharma-G/258_Final_Project/assets/107587243/badc01f9-f450-475d-a269-bddfa6e25867)

Model Registry is a means through which we can apply a version control on our models, it could be as simple as same model trained on data at 2 different times

![image](https://github.com/Samarth-Sharma-G/258_Final_Project/assets/107587243/d67576b5-92a8-42e5-9081-1bee41d0594e)

Endpoint Deployment allows us to expose our model as an API to the rest of the world.

![image](https://github.com/Samarth-Sharma-G/258_Final_Project/assets/107587243/4b0567f5-778f-4a81-90c6-d71f4a6660d0)

What we do is perform an experiment whenever new data comes in or as per the schedule, and it trains 3 models. The best performing model aka the champion model is registerd to the model registry. Then getting the model in the registry we perform A/B testing. Then finally the model is ready for deployement, but if there is already one model deployed, then the deployed model and the challenger model (the champion model from the experiment) are evaluated on the new data and if the challenger wins, it os deployed or the model already in production continues to serve.

## Inference Pipeline

![image](https://github.com/Samarth-Sharma-G/258_Final_Project/assets/107587243/addaed97-5f6b-456d-81f4-24fce42f2862)

As Endpoint and model are different, even if we deploy a different model the end point url remains the same so no changes there.

UI is fairly simple, you can upload an image from your device 

![image](https://github.com/Samarth-Sharma-G/258_Final_Project/assets/107587243/cfb56dd6-8999-4d2d-a4b1-71c980725a4c)

And a prediction will be returned:

![image](https://github.com/Samarth-Sharma-G/258_Final_Project/assets/107587243/abd6a07c-b2a1-4d11-af3e-b1271fa387b5)

![image](https://github.com/Samarth-Sharma-G/258_Final_Project/assets/107587243/eb78b458-310a-439a-b0bd-3e0deaf60a21)
