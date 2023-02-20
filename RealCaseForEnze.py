'''this script is used to help enze to use ANN for her own DGA data'''
##1.  data prepocessing (1: import data 2: data preprocessing(do we need normoalization?, do we need to split data as test and train data))
##2. train the kmeans model, get y_pred
##3. evaluate the performance of the kmeans on our data
    # evaluate analysis for each data
    # evaluate using accuracy score, precision score, recall score, and ROC figure
    # example: https://blog.csdn.net/gezigezao/article/details/105185061


## prepare the model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
## prepare data
import pandas as pd

'''train and test function'''
def train_test_DNN(x, y):
    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Normalize the input data
    scaler_x = StandardScaler()
    x_train = scaler_x.fit_transform(x_train)
    x_test = scaler_x.transform(x_test)

    # Scale the target variable
    scaler_y = MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.reshape(-1, 1))

    # Define the DNN architecture
    model = Sequential()
    model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    history = model.fit(x_train, y_train, epochs=200, batch_size=32, verbose=0, validation_data=(x_test, y_test))

    # Evaluate the model on the test set
    loss = model.evaluate(x_test, y_test, verbose=0)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Inverse transform the scaled predictions
    y_pred_orig = scaler_y.inverse_transform(y_pred)

    return model, loss, y_test, y_pred_orig, history











if __name__=='__main__':
    # prepare the control process parameter
    IF_TRAIN=True
    IF_SAVE_MODEL=True
    IF_LOAD_MODEL=True
    IF_PLOT_TRAIN=True
    # prepare the data
    df = pd.read_csv('data.csv')
    # replace Nan using 0
    df.fillna(0.0, inplace=True)
    x = df.iloc[:, :3].values
    y = df.iloc[:, -1].values
    # train and test model
    if IF_TRAIN:
        model, loss, y_test, y_pred_orig, history = train_test_DNN(x, y)
        model, loss, y_test, y_pred_orig, history = train_test_DNN(x, y)
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    if IF_SAVE_MODEL:
        model.save('enze_model.h5')
    if IF_LOAD_MODEL:
        model = load_model('enze_model.h5')
        scaler_x = StandardScaler()
        x = scaler_x.fit_transform(x)
        y_predict = model.predict(x)
        scaler_y = MinMaxScaler()
        get_fit = scaler_y.fit_transform(y.reshape(-1, 1))
        y_predict_ori=scaler_y.inverse_transform(y_predict)
        df['predicted_y'] = y_predict_ori
        df.to_csv('predicted_data.csv', index=False)



