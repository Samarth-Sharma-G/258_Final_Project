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



