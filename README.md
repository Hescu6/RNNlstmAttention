# RNNlstmAttention
Recurrent NN with attention
Encoder-Decoder Recurrent Neural Network with LSTM Attention
will be used to make predictions.
This model aims to evade sequential processing over time series with Attention 
as well as to mitigate the vanishing gradient problem seen in long range dependencies
by using LSTM and a generic attention module.

## Python
    o	time, numpy, time, and pyplot libraries
## Keras and TensorFlow libraries
    o	AttentionDecoder module (Ahmed, 2017)
## Jupyter Notebook
    o	matplotlib
## CSV Historical Data from Yahoo Finance)
    o	SPY - 26 years – weekly

In this project, Python is used as the programming language with the aid of numpy library to
help with calculations and array manipulation, the ‘matplotlib.pyplot‘ library is used to create
the charts in jupyter notebook, and the Keras library with Tensorflow backend to summon the neural network.

Keras library serves as an API that works with tensorflow to add the model’s hidden layers with LSTM cells.
Because at the time of this paper, Keras doesn’t have any module for ‘Attention’, a custom attention model
from a tutorial by Zafarali Ahmed (2017) is used.


# Steps in program
1.	Load Data into integer list
2.	Define batch size input and output for training
3.	One-hot encode and reshape list as 3D data
4.	Build RNN with LSTM and the Attention module
5.	Train LSTM NN
6.	Predict and one-hot decode data for output
7.	Plot prediction



# Parameters in Jupyter Notebook file
## Get Data
    o	Data_1: list obtained from csv file.
    o	Data_2: same list obtained from csv file, modified for testing purposes.
    o	Cardinality: total number of elements in the list.
    o	Maxval_train: highest number between data_1 and data_2 list.
## Set up model parameters
    o	Steps_in: Number of elements in epoch. Also serves as window size
    o	Steps_out: number of elements to be predicted
    o	Real: transforms the data_list 2 by shifting data steps_out spaces to give a more real time series prediction.
## Build model
    o	Model: coder-decoder RNN LSTM with attention model instance
## Train model
    o	Start: Starting point when traversing through the data. End-start = window size
    o	End: stop point for the data set to input into epoch. End-start = window size
    o	Minutes: seconds taken for training converted into minutes
    o	X and y: one hot encoded and 3-D transformed data_1 and data_2
## Predict
    o	Traverse: number of iterations needed to fully complete the prediction series
    o	Prediction: list where prediction is stored
    o	X and y: one hot encoded and 3-D transformed data_1 and data_2



# Functions in RNNattention.py file
## load_data
    o	Input: file name
    o	Output: Data list, its cardinality, and highest number
## encode
    o	Input: list and its highest value element
    o	One hot encodes data
    o	Output: encoded array
## decode
    o	Input: list
    o	One hot decodes data
    o	Output: decoded list
## regulate_window: 
    o	Parameters: start, end, window_step, cardinality, window size
    o	During training and predicting, this function regulates the window steps as well as the window size of the data to be fed in the model
    o	 Outputs the range that should be fed into the model as ‘start’ and ‘end’
## transform_data:
    o	Parameters: steps_in, steps_out, max_val, data_1, and data_2
    o	Calls encode function and reshapes list into 3D
    o	Output: encoded x and y 
## plot_it:
    o	Input: predicted data and true data
    o	Output: plotted data in graph

