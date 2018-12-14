import numpy as np
import theano
import theano.tensor as T
from keras import backend as K
import pdb

def custom_loss(y_true,y_pred):
    true_i = np.argmax(y_true)
    #pdb.set_trace()
    pred_i = np.arange(y_pred.shape)
    weights = K.variable(np.abs(pred_i-true_i))

    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss,-1)

    return loss


def get_weights(y_true, y_pred):
    true_i = np.argmax(y_true)
    pred_i = np.arange(y_pred.size)
    weights = np.abs(pred_i-true_i)   
    return weights

