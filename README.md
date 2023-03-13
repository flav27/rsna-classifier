# Train Code for RSNA Screening Mammography Breast Cancer Detection

https://www.kaggle.com/competitions/rsna-breast-cancer-detection

Dataset:

    Total unique patients: 11913
    Total unique images: 54706

For each patient there are multiple images of the L or R breast having MLO(mediolateral oblique) and CC(craniocaudal) views and 
The requirement is to predict the probability of cancer for each of the patients breasts.

- imbalanced dataset with only 1158 images containing cancer from the total of 54706.
- cancer size is very small relative to mammography

Data Preprocessing:
 - photometric-interpretation: as you can see Patient1 and Patient2 have different photometric-interpretations
 - windowing: also known as the "VOI-LUT" (Value of Interest-Look-Up Table) transform
 - normalize:
 - crop and resize

Patient1 Raw Images:

<img height="256" src="resources/Patient1Raw.PNG" width="1024"/>

Patient1 Processed Image:

<img height="256" src="resources/Patient1Processed.PNG" width="1024"/>

Patient2 Raw Images:

<img height="256" src="resources/Patient2Raw.PNG" width="1024"/>

Patient2 Processed Image:

<img height="256" src="resources/Patient2Processed.PNG" width="1024"/>



Neural Network Overview:

<img height="512" src="resources/Network Architecture.PNG" width="1024"/>




