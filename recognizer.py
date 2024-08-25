import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os


def get_model(shape):
    return Sequential([
        tf.keras.Input(shape=(shape,)),
        Dense(units=25, activation='relu'),
        Dense(units=15, activation='relu'),
        Dense(units=10, activation='linear')
    ])


def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    X_path = os.path.join(script_dir, "X.npy")
    y_path = os.path.join(script_dir, "y.npy")
    
    X = np.load(X_path)
    y = np.load(y_path)
    return X, y

def widgvis(fig):
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

def showResults(X, y, model):

    # You do not need to modify anything in this cell

    m, n = X.shape

    fig, axes = plt.subplots(8,8, figsize=(5,5))
    fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]
    widgvis(fig)
    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)
        
        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20,20)).T
        
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')
        
        # Predict using the Neural Network
        prediction = model.predict(X[random_index].reshape(1,400))
        prediction_p = tf.nn.softmax(prediction)
        yhat = np.argmax(prediction_p)
        
        # Display the label above the image
        ax.set_title(f"{y[random_index,0]},{yhat}",fontsize=10)
        ax.set_axis_off()
    fig.suptitle("Label, yhat", fontsize=14)
    plt.show()
