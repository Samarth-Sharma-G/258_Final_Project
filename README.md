# Wafer Defect Detection and MLOps

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

Baseline model was a good starting point. The model architecture was built by ChatGPT. Below is the performance:

![image](https://github.com/Samarth-Sharma-G/258_Final_Project/assets/107587243/b924007f-9d94-44b9-87cb-c3914273fa1f)

Observation:
- The model is overfitting the training data

Improvements to be incorporated:
- Regularization
- Improving model Architecture
- Increase the Data (recommended but not possible in this case)

Best Model's Performance after incorporating the changes:

![image](https://github.com/Samarth-Sharma-G/258_Final_Project/assets/107587243/aa427390-3989-4a64-8b10-4e281b9afc67)

Specific Changes:
- Using different regularization techniques such as Batch Normalization and Dropout layer
- Reducing the model complexity - specially after flattening the input
- Using newer more adaptive optimizer AdamW

Observation:

Our objective for model development phase was to create a model with 90+ performance (accuracy here), so we stop even though if we trained the model for more epoch the performance would have improved as the loss is still going down. 

Baseline vs Best Model Architectures:

![Baseline Model Architecture (1)](https://github.com/Samarth-Sharma-G/258_Final_Project/assets/107587243/1163b71e-ded1-45ac-bc7d-1f70fbd0976d)

## MLOps


