#p1
# Sequences & Recurrent Neural Networks
import numpy as np
# sequential data when order is important
# create a set of training or testing examples for the model, 
# where each example consists of an input of seq_length consecutive data points, 
# and the single target, the following data point.
def create_sequences(df, seq_length):
    xs, ys = [], []
    # Iterate over data indices
    for i in range(len(df) - seq_length):
      	# Define inputs
        x = df.iloc[i:(i+seq_length), 1] # column index 1 is consumption
        # Define target
        y = df.iloc[i+seq_length, 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)