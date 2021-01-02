This project focusses on the brain tumor detection using U-net. A dice score of .98 was achieved. The project is divided in 2 parts:
1). To train the u-net model using the file unet.py.
2). To segment the brain tumor region using predict.py.

Dataset:
The dataset contains 3064 .mat files with brain tumor images and corresponding masks. These images contain both HGG and LGG brain tumor images.
The total dataset is divided into training and test set in the ratio 3:1. The training set is placed in directory "Data/Train" and test set is placed at 
"Data/Test".

Working pipeline:
The working is as follows:
1). Using unet.py train the model with 50 epochs(epochs may be varied).
2). The model weights are automatically saved in a file called "result.h5
3). The predict.py file, when run, loads these trained and saved weights and then uses them to find the brain tumor region.
