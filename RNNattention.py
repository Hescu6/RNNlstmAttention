from keras.models import Sequential	#sequential model to build NN
from keras.layers import LSTM 	#used for long-short term memory cells
from attention_decoder import AttentionDecoder #attention module 
import matplotlib.pyplot as plt  #plot graph
import numpy as np  #used for arrays
import csv	#to read from file

file_1 = 'FB.csv'
file_2 = 'TQQQ.csv'

#returns data list, cardinality, and highest number
def load_data(filename):   
    temp = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            temp.extend(row)  
    result = [round(float(i)) for i in temp]
    return result, len(result), max(result) + 1

#One hot encode
def encode(data, num):
	result = []
	for i in data:
		temp = [0 for _ in range(num)]
		temp[i] = 1
		result.append(temp)
	return np.array(result)  #use numpy array to then be reshaped into 3d

 
#One hot decode
def decode(encoded_data):
	return [np.argmax(indx) for indx in encoded_data]


#Transform data (one-hot encoding and then reshaped to 3d) to be used in LSTM
def transform_data(el_in, el_out, max_val, data_1, data_2):
	x_train = data_1
	y_train = data_2[:el_out] + [0 for _ in range(el_in - el_out)] #assigns zeroes to elements > el_out
	#print ('******* SEQUENCE IN*****\n',sequence_in, '\n\n' )
	#print ('******* SEQUENCE OUT*****\n',sequence_out, '\n\n' )
	
	#one hot encode
	x = encode(x_train, max_val)
	y = encode(y_train, max_val)
	
	#reshape
	x = x.reshape((1, x.shape[0], x.shape[1])) #shape into 3d
	y = y.reshape((1, y.shape[0], y.shape[1])) #shape into 3d
	return x,y


 # regulate window so that it slides n steps at at time
 # and end of list is reached it starts back again at 0
def regulate_window(start, end, window_step, cardinality, window):
	
	#if cardinality - end < window_step:
	if end + window_step >= cardinality-1:
		if end == cardinality:
			return 0, window
		return cardinality - window, cardinality 
	return start + window_step, end + window_step
	


def predict_multiple(model, data, window_size, prediction_len):
    prediction_seqs = []
    for i in range(len(data)//prediction_len):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    print ('yo')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

def plot_it(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    ax.plot(predicted_data, label='Prediction')
    #Pad the list of predictions to shift it in the graph to it's correct start
    plt.legend()
    plt.show()

'''


#Load data from first file (etf or stock)
data_1, cardinality, maxval_1  = load_data(file_1)
#Load data from second file (etf or stock)
data_2, cardinality, maxval_2  = load_data(file_2)

print ('MAXVAL1 = ',maxval_1 - 1, '\n')
print ('MAXVAL2 = ',maxval_2 - 1, '\n')
print ('number of elements = ',cardinality, '\n')    
 
#print (data, '\n\n\n')


#Parameters
maxval_train = max(maxval_1, maxval_2) #fixed --- gets the largest element between both lists
steps_in = 90		#number of elements in batch for training
steps_out = 50	#number of elements to be predicted


# define model
model = Sequential()
model.add(LSTM(250, input_shape=(steps_in, maxval_train), return_sequences=True))
model.add(AttentionDecoder(250, maxval_train))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


# train LSTM
start = 0
end = steps_in
for epoch in range(1):
	
	x,y = transform_data(steps_in, steps_out, maxval_train, data_1[start:end], data_2[start:end])
	start, end = regulate_window(start, end, steps_in, cardinality)
	# fit model for one epoch on this sequence
	model.fit(x, y, epochs=1, verbose=2)


# evaluate LSTM
print ('NOW EVALUATE')
total = 1 
correct = 0
start = 0
end = steps_in
for _ in range(total):
	x,y = transform_data(steps_in, steps_out, maxval_train, data_1[start:end], data_2[start:end]) 
	y_test = model.predict(x, verbose=0)
	start, end = regulate_window(start, end, steps_in, cardinality)

	if np.array_equal(decode(y[0]), decode(y_test[0])):
		correct += 1
print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))


# spot check some examples
start = 0
end = steps_in
for _ in range(5):
	x,y = transform_data(steps_in, steps_out, maxval_train, data_1[start:end], data_2[start:end])
	y_test = model.predict(x, verbose=0)
	print('Expected:', decode(y[0]), '\nPredicted', decode(y_test[0]))
	start, end = regulate_window(start, end, steps_in, cardinality)
	
'''