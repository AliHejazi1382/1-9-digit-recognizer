import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
import os

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
    X_path = os.path.join(script_dir, "X.npy")
    y_path = os.path.join(script_dir, "y.npy")
    
    X = np.load(X_path)
    y = np.load(y_path)
    return X, y
