
# Weather Forecasting Project

This project is designed to forecast temperature and humidity using historical weather data. The project includes data preprocessing, model training, and forecasting scripts.

## Project Structure

- `2014-2024.csv`: The dataset containing historical weather data from 2014 to 2024, used for training and testing the models.
- `preprocess.py`: Script for preprocessing the data, including cleaning, normalization, and splitting into training and testing sets.
- `model.py`: Defines the machine learning models used for temperature and humidity forecasting.
- `train_temp.py`: Script to train the model specifically for temperature forecasting.
- `train_humidity.py`: Script to train the model specifically for humidity forecasting.
- `forecast_temp.py`: Script to generate temperature forecasts using the trained model.
- `forecast_humidity.py`: Script to generate humidity forecasts using the trained model.

## Installation

To run this project, you need to have Python installed along with the required dependencies. You can install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

*Note: Ensure that you have a `requirements.txt` file listing all the dependencies.*

## Usage

### Data Preprocessing

Before training the models, you need to preprocess the data:

```bash
python preprocess.py
```

This script will process the `2014-2024.csv` dataset and prepare it for model training.

### Training the Models

You can train the models for temperature and humidity forecasting separately:

- **Train Temperature Model:**

  ```bash
  python train_temp.py
  ```

- **Train Humidity Model:**

  ```bash
  python train_humidity.py
  ```

These scripts will save the trained models for future forecasting.

### Forecasting

After training the models, you can generate forecasts:

- **Forecast Temperature:**

  ```bash
  python forecast_temp.py
  ```

- **Forecast Humidity:**

  ```bash
  python forecast_humidity.py
  ```

These scripts will output the forecasted values based on the trained models.
