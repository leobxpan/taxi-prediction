from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dropout,Dense
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import regularizers
import numpy as np
import pdb
import datetime
import argparse

def main():
   
    global args

    parser = argparse.ArgumentParser(description="Main file for taxi prediction using LSTM")
    
    # ========================= Model Configs ========================== 
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--loss', type=str, default='mae', choices=['mse', 'mae'])

    # ========================= Learning Configs ==========================
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_steps', type=int, default=[20, 40], help='epochs to decay learning rate by 10')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4) 

    # ========================= Monitor Configs ==========================
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-freq', type=int, default=5)

    # ========================= Runtime Configs ==========================
    parser.add_argument('--val', type=bool, default=False)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--iter_p_epoch', type=int, default=500)

    args = parser.parse_args()
    
    # Data loading
    X_train = np.load('../assets/X_train.npy')
    y_train = np.load('../assets/y_train.npy')
    X_val = np.load('../assets/X_val.npy')
    y_val = np.load('../assets/y_val.npy')
    X_test = np.load('../assets/X_test.npy')
    y_test = np.load('../assets/y_test.npy')

    # Model configuration params
    feature_num = X_train.shape[2]
    sequence_len = X_train.shape[1]
    hidden_size = 128
    layer_num = 2

    # Model configurations
    model = Sequential()
    model.add(LSTM(units=hidden_size, input_shape=(sequence_len,feature_num), return_sequences=True))
    model.add(Dropout(args.dropout))
    for i in range(layer_num):
        model.add(LSTM(units=hidden_size, return_sequences=True))
        model.add(Dropout(args.dropout))
    model.add(LSTM(units=hidden_size,return_sequences=False))
    model.add(Dropout(args.dropout))
    model.add(Dense(1))

    # Optimization configuration
    model.compile(loss='mean_absolute_error' if args.loss=='mae' else 'mean_squared_error', optimizer='rmsprop')

    # Checkpoint setting
    checkpointer = ModelCheckpoint(filepath='./models/weights-{epoch:03d}-{val_loss:.4f}.hdf5', monitor='val_loss', save_best_only=True, verbose=1)
    hist = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(X_val, y_val), callbacks=[checkpointer],shuffle=True)
    print(hist.history)

    # Testing
    prediction = np.floor(model.predict(X_test, verbose=1)).flatten().tolist()
    loss = model.evaluate(X_test, prediction, batch_size=args.batch_size)
    error = np.subtract(y_test, prediction)
    relative_error = np.divide(error, y_test)
    print('error:{}'.format(error))
    mae = np.mean(np.fabs(error))
    print('mae:{}'.format(mae))

if __name__ == '__main__':
    main()
