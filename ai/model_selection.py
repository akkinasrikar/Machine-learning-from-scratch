import numpy as np
def train_test_split(x,y,test_size):
    shuffle_index_whole_set=np.random.permutation(len(x))
    x,y=x[shuffle_index_whole_set],y[shuffle_index_whole_set]
    x_length,y_length=len(x),len(y)
    training_length=int(x_length*(1.0-test_size))
    x_train,y_train,x_test,y_test=x[:training_length],y[:training_length],x[training_length:],y[training_length:]
    shuffle_index=np.random.permutation(training_length)
    x_train,y_train=x_train[shuffle_index],y_train[shuffle_index]
    return (np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test))