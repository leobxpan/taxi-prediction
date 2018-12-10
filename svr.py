from sklearn.svm import SVR
import numpy as np
import pdb

X_train = np.load('../assets/X_train.npy')
y_train = np.load('../assets/y_train.npy')
X_test = np.load('../assets/X_test.npy')
y_test = np.load('../assets/y_test.npy')

feature_num = X_train.shape[1]

model = SVR(C=1.0, epsilon=0.2)
model.fit(X_train, y_train)

prediction = model.predict(X_test)

mae = np.mean(np.fabs(prediction-y_test))
relative_mae = mae / np.mean(y_test)
print('mae:{}'.format(mae))
print('relative_mae:{}'.format(relative_mae))
