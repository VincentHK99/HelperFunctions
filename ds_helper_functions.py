#import relevent data science packages
from bs4 import BeautifulSoup
import requests
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

#####################################
### WEB SCRAPING HELPER FUNCTIONS ###
#####################################
"""
These functions should help streamline the process of building scraper algorithms. They should also help identify the meaning or response codes to aid debugging 
"""

"""
GET SOUP
"""

def get_soup(url,parser='html.parser',print_soup=False,timeout=10):
  """
    Sends a request to webpage and creates a soup object for a given url.
    If an error or a status code different to 200 occus the function will
    attemp to isolate the issue for the user of the function.
  """
  try:
    page = requests.get(url,timeout=timeout)
    
    # prove clarification as to what different status codes indicate
    if page.status_code == 200:
      print('Request successful!')
    elif page.status_code == 404:
      print('URL is not recognised')
    else:
      print(f'Uncommon response occured ({print(page.status_code)} refer to article below for quidance \n https://developer.mozilla.org/en-US/docs/Web/HTTP/Status')
    soup = BeautifulSoup(page.content,parser)
    if print_soup == True:
      print(soup.prettify())
  
  except:
    # the the user know if the requests package did not work
    print('An exception occurred. Perhaps the URL entered is incorrect')
    
##################################################
### TIME SERIES DEEP LEARNING HELPER FUNCTIONS ###
##################################################
"""
These functions should help build and analyse the quaility of time series forecast. Especially in tensorflow
"""
"""
MEAN ABSOLUTE SCALED ERROR
"""
    
def mean_absolute_scaled_error(y_true, y_pred):
 """
 Implement MASE (assuming no seasonality of data)
 """

 mae = tf.reduce_mean(tf.abs(y_true- y_pred))

 # find MAE of naive forecast by correctly indexing
 mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))

 return mae / mae_naive_no_season
  
  return soup

"""
PLOT TIME SERIES
"""
def plot_time_series(timesteps, values, format='-',start=0,end=None,label=None):
  """
  Plots a timesteps (a series of points in time) against values (a series of values across timesteps).
  
  Parameters
  ---------
  timesteps : array of timesteps
  values : array of values across time
  format : style of plot, default "."
  start : where to start the plot (setting a value will index from start of timesteps & values)
  end : where to end the plot (setting a value will index from end of timesteps & values)
  label : label to show on plot of values
  """
  # plot the time series dataset
  plt.plot(timesteps[start:end], values[start:end],format,label=label) #plot the seies based on the start and end periods in the function arguments
  plt.xlabel('Time')
  plt.ylabel('Revenue') 
  if label:
    plt.legend(fontsize=14) #if there is a legent set the font size to 14
  plt.grid(True)

"""
EVALUATE PREDICTIONS
"""
  
def evaluate_preds(y_true,y_pred):
  """
  This function will take y_true and y_prediction variables and use a variety of evaluation metrics to assess the performance of a time series algorithm 
  """
  # cast data to float32 datatype for metric calculations
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_preds = tf.cast(y_pred, dtype=tf.float32)

  # calculate the various metrics
  mae = tf.keras.metrics.mean_absolute_error(y_true,y_pred)
  mse = tf.keras.metrics.mean_squared_error(y_true,y_pred) #puts emphasis on outliers (they will heavily bias error term)
  rmse = tf.sqrt(mse)
  mape = tf.keras.metrics.mean_absolute_percentage_error(y_true,y_pred)
  mase = mean_absolute_scaled_error(y_true, y_pred) #as completed above

  return {"MAE": mae.numpy(),
          "MSE": mse.numpy(),
          "RMSE": rmse.numpy(),
          "MAPE": mape.numpy(),
          "MASE": mase.numpy()}


"""
GET LABELLED WINDOWS
"""
def get_labelled_windows(x, horizon=1):
  """
  Creates labels for windowed dataset.

  E.g. if horizon=1 (default)
  Input: [1, 2, 3, 4, 5, 6] -> Output: ([1, 2, 3, 4, 5], [6])
  """
  return x[:,:-horizon], x[:,-horizon:]

"""
MAKE WINDOWS
"""
def make_windows(x, window_size=7, horizon=1):
  """
  Turns a 1D array into a 2D array of sequential windows of window_size.
  """
  # 1. create a window of spececified size by adding the window size and horizon together
  window_step = np.expand_dims(np.arange(window_size+horizon),axis=0)
  print(f'Window step : \n {window_step}')

  # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
  window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)),axis=0).T
  print(f"Window indexes:\n {window_indexes[:3], window_indexes[-3:], window_indexes.shape}")

  # 3. index on the target array with 2D array of multiple window steps
  windowed_array = x[window_indexes]

  # 4. Get the labelled windows
  windows, labels = get_labelled_windows(windowed_array, horizon=horizon)

  return windows, labels
