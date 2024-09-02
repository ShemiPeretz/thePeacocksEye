import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def fill_missing_values(df):
    """
    Fills missing values in the DataFrame.

    Parameters:
    - df: Input DataFrame with potential missing values.

    Returns:
    - df: DataFrame with missing values filled.
    """
    df.replace('-', np.nan, inplace=True)

    # Fill missing values in numeric columns with the mean
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column] = df[column].fillna(df[column].mean())

    # Fill missing values in categorical columns with the mode
    for column in df.select_dtypes(include=[object]).columns:
        df[column] = df[column].fillna(df[column].mode()[0])

    return df

def date_csv(csv_file):
    """
    Reads and processes a CSV file containing weather data.

    Parameters:
    - csv_file: Path to the CSV file.

    Returns:
    - df: Processed DataFrame with datetime features and numeric columns converted.
    """
    df = pd.read_csv(csv_file, dtype=str, low_memory=False)
    df = fill_missing_values(df)

    numeric_columns = [
        'Temperature (°C)', 'Relative humidity (%)', 'Wind speed (m/s)', 'Grass temperature (°C)', 'Rainfall (mm)',
        'Pressure at station level (hPa)', 'Maximum temperature (°C)', 'Minimum temperature (°C)',
        'Wet Temperature (°C)', 'Wind direction (°)', 'Gust wind direction (°)', 'Maximum 1 minute wind speed (m/s)',
        'Maximum 10 minutes wind speed (m/s)', 'Gust wind speed (m/s)', 'Standard deviation wind direction (°)'
    ]

    # Convert numeric columns to proper numeric types
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')

    # Convert 'Date & Time (Summer)' column to datetime and drop rows with invalid dates
    df['Date & Time (Summer)'] = pd.to_datetime(df['Date & Time (Summer)'], format='%d/%m/%Y %H:%M', errors='coerce')
    df.dropna(subset=['Date & Time (Summer)'], inplace=True)

    # Extract year, month, day, and hour from the datetime column
    df['year'] = df['Date & Time (Summer)'].dt.year
    df['month'] = df['Date & Time (Summer)'].dt.month
    df['day'] = df['Date & Time (Summer)'].dt.day
    df['hour'] = df['Date & Time (Summer)'].dt.strftime('%H:%M')
    df.drop(columns=['Date & Time (Summer)'], inplace=True)

    return df

def mean_hour(df):
    """
    Computes the mean of the data for each hour.

    Parameters:
    - df: Input DataFrame with datetime features.

    Returns:
    - df_mean: DataFrame with mean values calculated for each hour.
    """
    df_copy = df.copy()

    # Group by hour and station (if available), then compute the mean for numeric columns
    if 'Station' in df_copy.columns:
        df_mean = df_copy.groupby(['Station', 'hour'], as_index=False).mean(numeric_only=True)
    else:
        df_mean = df_copy.groupby(['hour'], as_index=False).mean(numeric_only=True)

    return df_mean

class WeatherDataset(Dataset):
    """
    Custom Dataset class for weather data.

    Parameters:
    - X: Tensor containing input features.
    - y: Tensor containing target values.
    - sequence_length: Length of the sequences to be used in the LSTM model.
    """
    def __init__(self, X, y, sequence_length):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx):
        """
        Returns a tuple containing a sequence of input features and the corresponding target value.

        Parameters:
        - idx: Index for data retrieval.

        Returns:
        - (X_seq, y_target): Tuple containing the input sequence and target value.
        """
        X_seq = self.X[idx:idx + self.sequence_length]
        y_target = self.y[idx + self.sequence_length - 1]
        return X_seq, y_target

def preprocess_data(csv_file):
    """
    Preprocesses the weather data from the CSV file.

    Parameters:
    - csv_file: Path to the CSV file.

    Returns:
    - df: Preprocessed DataFrame with missing values filled and datetime features extracted.
    """
    df = date_csv(csv_file)
    df_mean = mean_hour(df)

    return df_mean

def prepare_data(df, target_col, test_size=0.2, sequence_length=24):
    """
    Prepares the data for training and testing by scaling, splitting, and converting to tensors.

    Parameters:
    - df: Input DataFrame with preprocessed data.
    - target_col: Column name of the target variable.
    - test_size: Proportion of data to be used for testing.
    - sequence_length: Length of the sequences to be used in the LSTM model.

    Returns:
    - train_dataset: Dataset object for training.
    - test_dataset: Dataset object for testing.
    - preprocessor: ColumnTransformer object for preprocessing.
    - scaler_target: StandardScaler object for scaling the target variable.
    - feature_names: List of feature names after preprocessing.
    """
    df = df.copy()

    # Define numeric and categorical columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns.remove(target_col)
    categorical_columns = df.select_dtypes(include=[object]).columns.tolist()
    feature_columns = numeric_columns + categorical_columns

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    # Fit and transform the features
    X = preprocessor.fit_transform(df[feature_columns])

    # Get feature names after preprocessing
    onehot_columns = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
        categorical_columns)
    feature_names = list(numeric_columns) + list(onehot_columns)

    # Scale target
    scaler_target = StandardScaler()
    y = scaler_target.fit_transform(df[[target_col]])

    # Convert to tensors
    X_tensor = torch.FloatTensor(X.toarray() if hasattr(X, "toarray") else X)
    y_tensor = torch.FloatTensor(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=test_size, shuffle=False
    )
    torch.save(X_train, f"X_train_{target_col}.pt")
    torch.save(X_test, f"X_test_{target_col}.pt")
    torch.save(y_train, f"y_train_{target_col}.pt")
    torch.save(y_test, f"y_test_{target_col}.pt")

    train_dataset = WeatherDataset(X_train, y_train, sequence_length)
    test_dataset = WeatherDataset(X_test, y_test, sequence_length)
    return train_dataset, test_dataset, preprocessor, scaler_target, feature_names

def main():
    """
    Main function for preprocessing the weather data and preparing it for training and testing.
    """
    csv_file = '2014-2024.csv'
    df = preprocess_data(csv_file)

    print("Columns in the DataFrame:")
    print(df.columns.tolist())

    for target_col in ['Temperature (°C)', 'Relative humidity (%)']:
        train_dataset, test_dataset, preprocessor, scaler_target, feature_names = prepare_data(df, target_col)

        print(f"\nPreprocessed data for {target_col}:")
        print(f"Number of features: {len(feature_names)}")
        print(f"Feature names: {feature_names}")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")

if __name__ == "__main__":
    main()
