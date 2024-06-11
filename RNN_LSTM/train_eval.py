
#Train the model
# Train the RNN model
rnn_model.fit(X, Y, epochs=100, batch_size=1, verbose=2)

# Train the LSTM model
lstm_model.fit(X, Y, epochs=100, batch_size=1, verbose=2)

#Evaluate the Model
# Make predictions
rnn_predictions = rnn_model.predict(X)
lstm_predictions = lstm_model.predict(X)

# Inverse transform the predictions to original scale
rnn_predictions = scaler.inverse_transform(rnn_predictions)
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# Evaluate the model
rnn_loss = rnn_model.evaluate(X, Y)
lstm_loss = lstm_model.evaluate(X, Y)

print(f'RNN Model Loss: {rnn_loss}')
print(f'LSTM Model Loss: {lstm_loss}')
