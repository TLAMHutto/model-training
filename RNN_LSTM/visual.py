import matplotlib.pyplot as plt

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(scaled_prices), label='Actual Prices')
plt.plot(np.arange(look_back, look_back + len(rnn_predictions)), rnn_predictions, label='RNN Predictions')
plt.plot(np.arange(look_back, look_back + len(lstm_predictions)), lstm_predictions, label='LSTM Predictions')
plt.legend()
plt.show()
