
##
## use in week 2, 4 and 5
##
import numpy as np

def get_numpy_data(data_sf, features, output):
    data_sf['constant'] = 1.0                # this is how you add a constant column to an SFrame
    features = ['constant', *features]       # add the column 'constant' to the front of the features list so that we can extract it...
    features_sf = data_sf[features]          # select the columns of data SFrame given by the features list (including constant):
    feature_matrix = features_sf.to_numpy()  # convert the features_SFrame into a numpy matrix:
    output_sarray = data_sf[output]          # assign the column of data_sf associated with the output to the SArray output_sarray
    output_array = output_sarray.to_numpy()  # convert the SArray into a numpy array by first converting it to a list
    return (feature_matrix, output_array)

def predict_output(feature_matrix, weights):
    ## assume feature_matrix is a numpy matrix containing the features
    ## as columns and weights is a corresponding numpy array
    return np.dot(feature_matrix, weights)

def get_rss(model, data, y):
    """
    get prediction, then calc. rss
    """
    preds = model.predict(data)   # First get the predictions
    diff = y - preds              # Then compute the residuals/errors
    rss = (diff * diff).sum()     # Then square and add them up
    return rss

def calc_rss(data, y, weights):
  preds = predict_output(data, weights)
  return np.sum(np.square(preds - y))
