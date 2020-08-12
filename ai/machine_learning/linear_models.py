import numpy as np
class gradient_descent:

    def fit(self,x_train,y_train,learning_rate,epochs):
        self.m=np.random.randn(1,x_train.shape[1])*1000
        self.c=500
        l=len(x_train)
        for i in range(epochs):
            intercept=np.sum(np.subtract((np.dot(self.m,x_train.T)+self.c),y_train))
            slope=sum(np.dot(((np.dot(self.m,x_train.T)+self.c)-y_train),x_train))
            self.c=self.c-np.dot(learning_rate,intercept)/l
            self.m=self.m-np.dot(learning_rate,slope)/l
    def slope_intercept(self):
        print(self.m,self.c)
    def predict(self,x_test):
        self.m=self.m.reshape(self.m.shape[1],self.m.shape[0])
        result=np.dot(x_test,self.m)+self.c
        return result

class SGD:

    def fit(self,x_train,y_train,learning_rate,epochs):
        self.m=np.random.randn(1,x_train.shape[1])
        self.c=0.5
        l=len(x_train)
        for i in range(epochs):
            intercept=0
            slope=0
            for j in range(l):
                ind=np.random.randint(l)
                s=x_train[ind].reshape(x_train[ind].shape[0],1)
                intercept=np.sum(np.subtract((np.dot(self.m,s)+self.c),y_train[ind]))
                slope=sum(np.dot(((np.dot(self.m,s)+self.c)-y_train[ind]),s.T))
            self.c=self.c-np.dot(learning_rate,intercept)/l
            self.m=self.m-np.dot(learning_rate,slope)/l
    def slope_intercept(self):
        print(self.m,self.c) 
               
    def predict(self,x_test):
        self.m=self.m.reshape(self.m.shape[1],self.m.shape[0])
        result=np.dot(x_test,self.m)+self.c
        return result

class Linear_regression:

    def fit(self,x_train,y_train):
        x_new=np.c_[np.ones((len(x_train),1)),x_train]
        self.m_c=np.linalg.inv(x_new.T.dot(x_new)).dot(x_new.T).dot(y_train)
        self.m=self.m_c[1:]
        self.c=self.m_c[0:1]

    def predict(self,x_test):
        x_new_test=np.c_[np.ones((len(x_test),1)),x_test]
        y_pred=x_new_test.dot(self.m_c)
        return y_pred




        


