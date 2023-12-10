# Romania Weather Forecast using RNNs

Implementing RNNs to predict weather forecasting using data from a meterological station in Romania.

## Table of contents

1. [Environment setup](#environment)
2. [Dataset creation and description](#dataset)
3. [Next day temperature prediction](#next-day-temperature-prediction)
4. [Extended future temperature prediction](#predicting-the-future)

## Environment

### Setup

In order to run the scripts, an installation of Python is required. Preferably **Python 3.8.10**

    $ python -V
    Python 3.8.10

Once Python is installed, the project environment dependencies should be installed.

    # Create a virtual environment
    $ python -m venv venv

    # Activate virtual environment
    $ . venv/bin/activate

    # Install project dependencies
    (venv) $ pip install -r requirements.txt

## Dataset

### Creation

To obtain the dataset of meteorological data used for training, you can run the `get_data.py` script in the `src/` folder that automatically creates a `.csv` file in the `datasets/` folder.

The dataset creation script scrapes data from [tutiempo](https://en.tutiempo.net/climate/romania/2/) using [Selenium](https://www.selenium.dev/), for the given time period, for the specified weather station. This creates a Chrome web driver that automatically visits pages and extracts data into a csv file.

Steps to run:

1.  Set environment variables in `env/data.env`

        # env/data.env
        WEATHER_STATION="154210" # Bucuresti Otopeni
        START_YEAR="2017"
        END_YEAR="2022"

2.  Run the script from the parent folder (where this `README` is located)

        (venv) $ python src/get_data.py

### Description

Depending on the time period specified by the user, the duration of the script may vary. After finishing, the created csv file will have the following columns:

| Column              | Description                                 |
| ------------------- | ------------------------------------------- |
| Year                |                                             |
| Month               |                                             |
| Day                 |                                             |
| Temperature_Avg     | Average Temperature (°C)                    |
| Temperature_Max     | Maximum temperature (°C)                    |
| Temperature_Min     | Minimum temperature (°C)                    |
| Sea_Level_Pressure  | Atmospheric pressure at sea level (hPa)     |
| Humidity_Avg        | Average relative humidity (%)               |
| Precipitation_Total | Total rainfall and / or snowmelt (mm)       |
| Visibility_Avg      | Average visibility (Km)                     |
| Wind_Avg            | Average wind speed (Km/h)                   |
| Wind_Sustained_Max  | Maximum sustained wind speed (Km/h)         |
| Wind_Max            | Maximum speed of wind (Km/h)                |
| Is_Rain             | Indicates whether there was rain or drizzle |
| Is_Snow             | Indicates whether there was snow            |
| Is_Storm            | Indicates whether there was storm           |
| Is_Fog              | Indicates whether there was fog             |

Example of the csv file created:

    Year,Month,Day,Temperature_Avg,Temperature_Max,Temperature_Min,Sea_Level_Pressure,Humidity_Avg,Precipitation_Total,Visibility_Avg,Wind_Avg,Wind_Sustained_Max,Wind_Max,Is_Rain,Is_Snow,Is_Storm,Is_Fog
    2017,01,1,-2,5,-6,,76,0,6,12.8,18.3,25.2,False,False,False,False
    2017,01,2,-1.6,6.3,-6,,71,0,7.2,12.6,24.1,28.7,False,False,False,False
    2017,01,3,-1.3,5.2,-7,,71,0,6.9,7.2,18.3,21.7,False,False,False,False
    2017,01,4,-2,4.6,-7.3,,75,0,6,11.3,22.2,25.2,False,False,False,False

## Next day temperature prediction

### Training

For this project, 2 different RNN types were tried, LSTM and GRU. The model definitions can be found in the `src/models.py` file.

To train these models for next day temperature prediction, the user has to follow these steps:

1.  Select minimum and maxim temperature values for scaling the dataset in `env/dataset.env`

        # env/dataset.env
        MIN_TEMP="-35"
        MAX_TEMP="45"

2.  Select a training learning rate and dropout rate in `env/train.env`

        # env/train.env
        # RNN parameters
        DROPOUT=0.4

        # Train parameters
        LEARNING_RATE=0.001

3.  Run the training script

        (venv) $ python src/train.py

This script trains each of the 2 models with a varying number of hyperparameters, as presented in the following table. Each model will be saved in the `models/` folder, with a folder structure matching the hyperparameters used to train the given model. Example folder structure:

```
models
├── GRU
│   └── lags
│       ├── 3
│       │   └── hidden
│       │       ├── 128
│       │       │   └── layers
│       │       │       ├── 2
│       │       │       │   └── dropout
│       │       │       │       └── 0.4
│       │       │       │           └── learning_rate
│       │       │       │               └── 0.001
│       │       │       │                   └── epochs
│       │       │       │                       ├── 100
│       │       │       │                       │   ├── loss_curves.jpg
│       │       │       │                       │   ├── params.pt
│       │       │       │                       │   └── validation_performance.jpg
│       │       │       │                       ├── 150
│       │       │       │                       │   ├── loss_curves.jpg
│       │       │       │                       │   ├── params.pt
│       │       │       │                       │   └── validation_performance.jpg
│       │       │       │                       └── 50
│       │       │       │                           ├── loss_curves.jpg
│       │       │       │                           ├── params.pt
│       │       │       │                           └── validation_performance.jpg
│       │       │       ├── 4
│       │       │       │   └── dropout
│       │       │       │       └── 0.4
│       │       │       │           └── learning_rate
│       │       │       │               └── 0.001
│       │       │       │                   └── epochs
│       │       │       │                       ├── 100
│       │       │       │                       │   ├── loss_curves.jpg
│       │       │       │                       │   ├── params.pt
│       │       │       │                       │   └── validation_performance.jpg
...     ...     ...     ...                     ...
```

| Model Type | No. time lags | No. features in the hidden state | No. layers | No.epochs |
| ---------- | ------------- | -------------------------------- | ---------- | --------- |
| LSTM       | 3             | 32                               | 2          | 50        |
| GRU        | 5             | 64                               | 4          | 100       |
|            | 7             | 128                              | 8          | 150       |

Because we have 162 models to train (2x3x3x3x3) the training script can take a very long time. for a Nvidia RTX 3050 it took approximately 20 hours to train, each model being trained in approx 10 minutes.

### Model Evaluation

Evaluation is done automatically in the training script for each model, saving 2 graphs in the model paths, one representing the training loss curves `loss_curves.jpg` and the other representing the validation performance of the trained model `validation_performance.jpg`.

Furthermore, the RMSE, MAPE and R score of each model is saved in a `results.csv` file in the same folder as this `README`, which can be used to evaluate the trained models. An example of the `results.csv` is present in the current repository.

## Predicting the future

Extended future prediction is the process through which a model bases its future predictions on its past predictions, instead of the actual real-world values. In order to train a RNN model for extended future prediction, one must follow these steps:

1.  Select the model training parameters in the `env/future.env` file

        # env/future.env
        MODEL_TYPE="GRU"
        INDEX=3
        LAGS=3
        HIDDEN=32
        LAYERS=2
        DROPOUT=0.4
        LEARNIG_RATE=0.001
        EPOCHS=150

2.  Specify a train and test dataset path from the `datasets/` folder. Could be obtained by running the `src/get_data.py` script with different `env/data.env` parameters (see [dataset creation](#creation)).

        # env/future.env
        TRAIN_DS_PATH="datasets/154210_2017_2022.csv"
        TEST_DS_PATH="datasets/154210_2023_2023.csv"

3.  Run the prediction script

        (venv) $ python src/predict_future.py

This will save a trained model in the `models/future/` folder, together with the model training loss curves as well as a vizualization of the prediction performance on the test set. This script will also save the RMSE, MAPE and R values in a `future.csv` file in the current folder, for analysis purposes.
