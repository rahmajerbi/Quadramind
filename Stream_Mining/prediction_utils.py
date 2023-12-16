# prediction_utils.py

import numpy as np

def predict_future(model, X, n_steps):
    predictions = []
    current_input = X[-1].reshape((1, X.shape[1], X.shape[2]))

    for _ in range(n_steps):
        next_point_scaled = model.predict(current_input, verbose=0)
        next_point = next_point_scaled.flatten()
        predictions.append(next_point)
        current_input = np.roll(current_input, shift=-1, axis=1)
        current_input[0, -1, :] = next_point_scaled[0]

    return predictions
