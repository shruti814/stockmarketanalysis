import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def plot_forecast(dates, y_true, preds, labels=None, title="Actual vs Forecast"):
    plt.figure(figsize=(10,5))
    plt.plot(dates, y_true, label='Actual')
    for p,label in zip(preds, labels):
        plt.plot(dates, p, label=label)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    return plt
