import numpy as np
def mean_square_error(y_actual,y_pred):
    n=len(y_pred)   
    error=(np.sum((y_actual-y_pred)**2))/n
    return error

def root_mean_square_error(y_actual,y_pred):
    n=len(y_pred)  
    error=(np.sum((y_actual-y_pred)**2))/n
    return np.sqrt(error)

def _mod(number):
    s=0
    for i in number:
        if i<0:
            number[s]=-i
        else:
            number[s]=i
        s=s+1
    return number

def mean_absolute_error(y_actual,y_pred):
    n=len(y_pred)  
    error=(np.sum(_mod(y_actual-y_pred)))/n
    return error

def explained_variance_score(y_actual,y_pred):
    y_actual=y_actual.reshape(y_actual.shape[0],1)
    shape1,shape2=y_actual.shape[0],y_actual.shape[1]
    y_actual=y_actual.reshape(shape2,shape1)
    y_pred=y_pred.reshape(shape2,shape1)
    error=1-((np.cov(y_actual-y_pred))/np.cov(y_actual))
    return error

def r2_score(y_actual,y_pred):
    n=len(y_actual)
    m=len(y_actual)
    sup=np.sum((y_actual-y_pred)**2)
    sdown=np.sum((y_actual-y_actual.mean())**2)
    error=1-(sup/sdown)
    return error

