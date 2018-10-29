## Libraries
import pandas as pd
import numpy as np
import gensim
from gensim.models import KeyedVectors
import gensim.models as Word2Vec
import gensim.downloader as api
model_keyedVectors = api.load("word2vec-google-news-300")
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
# ML libraries
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
# DL libraries
from keras.layers import Convolution1D, Flatten, Dropout, Dense, MaxPool1D, MaxPooling1D, GlobalMaxPooling1D, Convolution2D, Conv2D, Conv1D, Activation, BatchNormalization
from keras.layers import LSTM, Bidirectional, Concatenate, Input, concatenate
from keras.layers.embeddings import Embedding
from keras.models import Sequential, Model
## OHE for ylabels
from keras.utils import np_utils
## optimizer import
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier


# read in data 
df = pd.read_csv("full_sentences.csv",index_col = 0)
df = df[~df['Sentence'].isna()]

df.reset_index(drop = True, inplace = True)

df['recognised_words'] = df['Sentence'].apply(lambda x: [word for word in word_tokenize(x) if word in model_keyedVectors.wv.vocab and word not in stopwords.words('english')])

X_data = df['recognised_words']
y_output = df['Sentiment']

## self-defined functions 
def helper_function(_word):
    if len(_word) != 0:
        return model_keyedVectors.get_vector(_word)
    else:
        return np.zeros(300)
      
def generate_wordEmbeddings(_maxSentenceLength,_numClasses,_Xtrain,_Xtest,_ytrain,_ytest):
    """
    input: sentence length to pad till, number of classes in our dataset, X training, X validation , y training, y validation
    function: to generate the word embeddings into list form
    output: processed xtrain, xtest, ytrain, ytest for training and validation
    """
    _Xtrain_padded = _Xtrain.apply(lambda l: ([''] * (_maxSentenceLength - len(l)) + l))
    _Xtest_padded = _Xtest.apply(lambda l: ([''] * (_maxSentenceLength - len(l)) + l))
    ##uses the helper function to form a list of word embeddings from the recognised words 
    _Xtrain_padded = _Xtrain_padded.apply(lambda x: np.array([helper_function(word) for word in x])).tolist()
    _Xtest_padded = _Xtest_padded.apply(lambda x: np.array([helper_function(word) for word in x])).tolist()
    _Xtrain_padded_array = np.array(_Xtrain_padded)
    _Xtest_padded_array = np.array(_Xtest_padded)
    ## OHE for y labels
    _ytrain = np_utils.to_categorical(_ytrain, _numClasses)
    _ytest = np_utils.to_categorical(_ytest, _numClasses)
    return _Xtrain_padded_array, _Xtest_padded_array, _ytrain, _ytest

################### First model using only LSTM ################################
def RNN_model(_LSTMunit,_inputShape,_loss,_optimizer,_classes):
    """
    inputs: the first lstm layer's unit, input shape, loss , optimizer, number of classes to identify in the dataset
    function: to create the sequential model architecture using LSTM
    output: compiled keras model
    """
    model_RNN= Sequential()
    model_RNN.add(Bidirectional(LSTM(_LSTMunit,return_sequences=True),input_shape = _inputShape))
    model_RNN.add(Bidirectional(LSTM(10)))
    model_RNN.add(Dense(_classes))
    model_RNN.add(Activation('softmax'))
    model_RNN.compile(loss=_loss, optimizer=_optimizer,metrics=['accuracy'])
    return model_RNN
  
# model_RNN = RNN_model(300,inputShape,'categorical_crossentropy','rmsprop',3)

################# Second model using only CNN layers and pooling ####################
def CNN_model(_inputShape, _filterSize, _KernelSize, _loss, _optimizer,_classes):
    """
    inputs: inputshape, filter size for the first layer, kernel size for the first layer, loss (usually 'categorical_crossentropy'), optimizer = Adam(), num of classes
    function: to create sequential model for cnn model only
    output: compiled CNN model
    """
    model_CNN = Sequential()
    model_CNN.add(Conv1D(filters=_filterSize,kernel_size=_KernelSize, input_shape = _inputShape))
    model_CNN.add(MaxPooling1D(2))
    model_CNN.add(Conv1D(128,3))
    model_CNN.add(MaxPooling1D(2))
    model_CNN.add(Conv1D(128,2))
    model_CNN.add(MaxPooling1D(5))
    model_CNN.add(Flatten())
    model_CNN.add(Dense(_classes, activation='softmax')) ## activation layer
    ## compile 
    model_CNN.compile(loss=_loss, optimizer=_optimizer, metrics=['accuracy'])
    return model_CNN

# model_cnn = CNN_model(inputShape, 128,2,'categorical_crossentropy',Adam(),3)

######################### Third model using multi-input CNN model  ############################
def multiInput_CNN_model(_inputShape,_kernelSizeList,_filterSize, _loss, _optimizer,_classes):
    """
    References: # https://github.com/keras-team/keras/issues/6547  , # https://arxiv.org/pdf/1510.03820.pdf, # https://www.programcreek.com/python/example/89660/keras.layers.concatenate
    input: inputshape, list of kernel sizes for forming the multi input CNN layers, filter size for the first layer, loss = ('categorical_crossentropy'), optimiser = Adam(), num of classes
    function: uses the Functional API from keras to form a multi input CNN model. The different starting inputs are for different kernel size of 2,3,4 which will slide over the words 
            in the sentence to capture the n-gram sematics.
    output: compiled multi-input CNN model
    """
    input_sub = Input(_inputShape)
    submodels = []
    for k in _kernelSizeList:
        ## create a convolutional layer for the input for each of the kernel size
        ## input shape of (no.samples, sentence length, embedding feature)
        output_sub_1 = Conv1D(filters = _filterSize, kernel_size = k)(input_sub)
        output_sub_2 = BatchNormalization()(output_sub_1)
        output_sub_3 = Activation("relu")(output_sub_2)
        output_sub_4 = GlobalMaxPooling1D()(output_sub_3)
        ## append into the list
        submodels.append(output_sub_4)
    ## outside of the loop, concatenate the layers together 
    outer_output = concatenate(submodels,axis = 1)
    outer_output_1 = Dense(128, activation = 'relu')(outer_output)
    ## activation layer 
    outer_output_2 = Dense(_classes, activation = "softmax")(outer_output_1)
    model_multiInput = Model(inputs = input_sub, outputs= outer_output_2)
    model_multiInput.compile(loss= _loss, optimizer= _optimizer, metrics=['accuracy'])
    return model_multiInput

# model_multiInput = multiInput_CNN_model(inputShape, [2,3,4],300,'categorical_crossentropy',Adam(),3)
## slight improvement from the purely one input layer cnn model but there is clearly overfitting because training acc = 0.999 >>>>> val accuracy.
## to try:
# 1. to impose dropout layers + chnge number of filters = 2 
# 2. to input RNN models to take sequence into account together ? 

##################### Fourth model(s) with both multi-input CNN and LSTM ################################
## similar architecture with different complexity
#1 - simpler model
def multiInput_CNN_RNN_model_simple(_inputShape, _kernelSizeList, _ConvFilterSize, _LSTMUnits,_loss,_optimizer,_classes):
    """
    Reference:# https://medium.com/@sabber/classifying-yelp-review-comments-using-cnn-lstm-and-visualize-word-embeddings-part-2-ca137a42a97d
    inputs: input shape, list of kernel sizes, conv filter size for first layer, lstm units for the only rnn layer, loss = 'categorical_crossentropy', optimizer = Adam(), num of classes
    function: an extension of the cnn and rnn model with less layers after merging the input layers 
    output: compiled model
    """
    input_sub = Input(_inputShape)
    submodels = []
    for k in _kernelSizeList:
        ## create a convolutional layer for the input for each of the kernel size
        ## input shape of (no.samples, sentence length, embedding feature)
        output_sub_1 = Conv1D(filters = _ConvFilterSize, kernel_size = k)(input_sub) # Filter size here can change 2 or 128? research paper uses 2 for each kernel size
        output_sub_2 = BatchNormalization()(output_sub_1)
        output_sub_3 = Activation("relu")(output_sub_2)
        output_sub_4 = MaxPooling1D()(output_sub_3)
        output_sub_5 = LSTM(_LSTMUnits)(output_sub_4)  ## to try different amount for this 
        submodels.append(output_sub_5)
    ## outer layer concatenating
    outer_output = concatenate(submodels,axis = 1)
    ## activation layer
    outer_output_1 = Dense(_classes)(outer_output)
    main_output = Activation('softmax')(outer_output_1)
    model_both_w_CNN_RNN_simple = Model(inputs = input_sub, outputs= main_output)
    model_both_w_CNN_RNN_simple.compile(loss=_loss, optimizer=_optimizer, metrics=['accuracy'])
    return model_both_w_CNN_RNN_simple

# model_simple = multiInput_CNN_RNN_model_simple(inputShape,[2,3,4],2,128,'categorical_crossentropy',Adam(),3)

#2 - complex-er model
def multiInput_CNN_RNN_model_complex(_inputShape, _kernelSizeList, _ConvFilterSize, _LSTMUnits,_loss,_optimizer,_classes):
    """
    Reference:# https://medium.com/@sabber/classifying-yelp-review-comments-using-cnn-lstm-and-visualize-word-embeddings-part-2-ca137a42a97d
    inputs: input shape, list of kernel sizes, conv filter size for first layer, lstm units for the only rnn layer, loss = 'categorical_crossentropy', optimizer = Adam()
    function: an extension of the cnn and rnn model with more layers after merging the input layers 
    output: compiled model
    """
    input_sub = Input(_inputShape)
    submodels = []
    for k in _kernelSizeList:
        ## create a convolutional layer for the input for each of the kernel size
        ## input shape of (no.samples, sentence length, embedding feature)
        output_sub_1 = Conv1D(filters = _ConvFilterSize, kernel_size = k)(input_sub) # Filter size here can change 2 or 128? research paper uses 2 for each kernel size
        output_sub_2 = BatchNormalization()(output_sub_1)
        output_sub_3 = Activation("relu")(output_sub_2)
        output_sub_4 = MaxPooling1D()(output_sub_3)
        output_sub_5 = LSTM(_LSTMUnits)(output_sub_4)  ## to try different amount for this 
        submodels.append(output_sub_5)
    ## outer layer concatenating
    outer_output = concatenate(submodels,axis = 1)
    outer_output_1 = Dropout(0.5)(outer_output)
    outer_output_2 = Dense(128, activation = None)(outer_output_1)
    outer_output_3 = BatchNormalization()(outer_output_2)
    outer_output_4 = Activation('relu')(outer_output_3)
    outer_output_5 = Dropout(0.25)(outer_output_4)
    outer_output_6 = Dense(128, activation = None)(outer_output_5)
    outer_output_7 = BatchNormalization()(outer_output_6)
    outer_output_8 = Activation('relu')(outer_output_7)
    outer_output_9 = Dense(3)(outer_output_8)
    outer_output_10 = BatchNormalization()(outer_output_9)
    ## activation layer
    outer_output_11 = Dense(_classes)(outer_output_10)
    main_output = Activation('softmax')(outer_output_11)
    model_both_w_CNN_RNN = Model(inputs = input_sub, outputs= main_output)
    model_both_w_CNN_RNN.compile(loss=_loss, optimizer=_optimizer, metrics=['accuracy'])
    return model_both_w_CNN_RNN

# model_both_complex = multiInput_CNN_RNN_model_complex(inputShape, [2,3,4],2,128,'categorical_crossentropy',Adam(),3)

############################################ Analysis of results obtained ################################################

## random thoughts. to be confirmed with research
##Accuracy are better for the multi input cnn model. Perhaps word sematic are better captured using CNN? Accuracy ##lies around 0.5 - 0.65




## implementing gridsearchCV to run in remote server in finding the best parameters for RNN model 
# reference:
# https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
epochs_list = [10,20,50,70,100,200]
batchSizes_list = [10,20,30,50]
optimizers_list = ['rmsprop','Adam','SGD','Adagrad']
LSTM_layers_units_lists = [100,200,300]
param_grid = dict(batch_size=batchSizes_list, epochs=epochs_list,optimizer = optimizers_list, LSTMunit = LSTM_layers_units_lists )


X_data_padded = X_data.apply(lambda l: ([''] * (50 - len(l)) + l))
##uses the helper function to form a list of word embeddings from the recognised words 
X_data_padded_list = X_data_padded.apply(lambda x: np.array([helper_function(word) for word in x])).tolist()
X_data_padded_array = np.array(X_data_padded_list)
## OHE for y labels
y_output_OHE = np_utils.to_categorical(y_output, 3)

inputShape = (X_data_padded_array.shape[1],X_data_padded_array.shape[2])

def create_model(LSTMunit = 300,optimizer = 'rmsprop'):
    model = RNN_model(LSTMunit,inputShape,'categorical_crossentropy',optimizer,3)
    return model

# RNN model - need an additional layer of abstraction for gridsearchcv to work
model = KerasClassifier(build_fn=create_model, verbose=1)

grid = GridSearchCV(estimator=model,param_grid=param_grid,n_jobs = 1,cv= 10)

grid_result = grid.fit(X_data_padded_array, y_output_OHE)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



