import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

data = pd.read_csv("dataset1.csv")

endog = data['Amount']
p, d, q = 0, 1, 0

# Create and fit the ARIMA model
arima_model = ARIMA(endog, order=(p, d, q))
arima_model_fit = arima_model.fit()

# Make predictions
predictions = arima_model_fit.predict(start=100, end=200)
print(predictions)

# Calculate Mean Absolute Error (MAE)
test_data = endog[100:201]  # Assuming you have a separate test dataset
print(test_data)
mae = np.mean(np.abs(predictions - test_data))
print("Mean Absolute Error (MAE): ", mae)
mape = np.mean(np.abs((test_data - predictions) / test_data))
print("Mean Absolute Percentage Error (MAPE):",mape,"%")


# Plot the actual values
plt.plot(test_data.index, test_data.values, label='Actual')

# Plot the predicted values
plt.plot(predictions.index, predictions.values, label='Predicted')

plt.xlabel('Past')
plt.ylabel('Traffic')
plt.title('ARIMA Model Forecast')
plt.legend()
plt.show()
