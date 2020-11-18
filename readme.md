Code for the Paper "Diffusion Models for Handwriting Generation": https://arxiv.org/abs/2011.06704

Project written in python 3.7, Tensorflow 2.3

To run the model, install required packages with 
pip install -r requirements.txt 

Then run inference.py and specify arguments as needed


To retrain model, run train.py, and specify arguments to change hyperparameters
All models will be saved in the ./weights directory


Before running the training script, download the following things from 
https://fki.tic.heia-fr.ch/databases/download-the-iam-on-line-handwriting-database

data/lineStrokes-all.tar.gz   -   the stroke xml for the online dataset
data/lineImages-all.tar.gz    -   the images for the offline dataset
ascii-all.tar.gz              -   the text labels for the dataset
extract these contents and put them in the ./data directory 
