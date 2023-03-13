# Train Code for RSNA Screening Mammography Breast Cancer Detection

https://www.kaggle.com/competitions/rsna-breast-cancer-detection

"**Goal of the Competition**

The goal of this competition is to identify breast cancer. You'll train your model with screening mammograms obtained 
from regular screening.

Your work improving the automation of detection in screening mammography may enable radiologists to be more accurate and
efficient, improving the quality and safety of patient care. It could also help reduce costs and unnecessary medical 
procedures."

Dataset:

    Total unique patients: 11913
    Total unique images: 54706

For each patient there are multiple images of the L or R breast having MLO(mediolateral oblique) and CC(craniocaudal) views.
The prediction required can be summarized as this: for each patient predict the probability of cancer for each of the
breasts.

A challenging aspect of the competition was that the dataset was imbalanced with only 1158 images containing cancer from
the total of 54706.

Another challenging aspect was that the cancer size can be very small from mm to 2-3 cm in diameter max, and the scanned images are large, 
with sizes around 5000px.

I have applied data preprocessing with the following:
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



