import numpy as np
class gradient_descent:

    def fit(self,x_train,y_train,learning_rate,epochs):
        self.m=np.random.randn(1,x_train.shape[1])*1000
        self.c=500
        l=len(x_train)
        for i in range(epochs):
            intercept=np.sum(np.subtract((np.dot(self.m,x_train.T)+self.c),y_train))
            slope=sum(np.dot(((np.dot(self.m,x_train.T)+self.c)-y_train),x_train))
            #print(slope,intercept)
            self.c=self.c-np.dot(learning_rate,intercept)/l
            self.m=self.m-np.dot(learning_rate,slope)/l
            #print(m,c)
        #print(m,c)
    def predict(self,x_test):
        self.m=self.m.reshape(self.m.shape[1],self.m.shape[0])
        result=np.dot(x_test,self.m)+self.c
        return result

class Linear_regression:

    def fit(self,x_train,y_train):
        x_new_train=np.c_[np.ones((len(x_train),1)),x_train]
        self.weigts_and_intercept=np.linalg.pinv(x_new_train.T.dot(x_new_train)).dot(x_new_train.T).dot(y_train)
        self.weights=self.weigts_and_intercept[1:]
        self.intercept=self.weigts_and_intercept[0:1]

    def predict(self,x_test):
        x_new_test=np.c_[np.ones((len(x_test),1)),x_test]
        y_pred=x_new_test.dot(self.weigts_and_intercept)
        return y_pred


        


