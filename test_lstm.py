from keras.models import load_model
import numpy as np
import pdb

# Parameter declaration
batch_size = 128

# Load data
X_test = np.load('../assets/X_test.npy')
y_test = np.load('../assets/y_test.npy')

# Load model
model = load_model('./models/weights-050-2.8417.hdf5')

# Prediction
prediction = model.predict(X_test, verbose=1)
loss = model.evaluate(X_test, prediction, batch_size=batch_size)
prediction = np.floor(prediction).flatten().tolist()

error = np.subtract(y_test, prediction)
relative_error = np.divide(error, y_test)
print 'error:', error
print 'relative_error:', np.mean(np.fabs(relative_error))
mse = np.mean(np.square(error))
print 'mse:', mse
mae = np.mean(np.fabs(error))
print 'mae:',mae
