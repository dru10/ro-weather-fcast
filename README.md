# Romania Weather Forecast using RNNs

Implementing RNNs to predict weather forecasting using data from a meterological station in Romania.

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

To obtain the dataset of meteorological data used for training, you can run the `get_data.py` script that automatically creates a `data.csv` file in the current folder.

The dataset creation script scrapes data from [tutiempo](https://en.tutiempo.net/climate/romania/2/) using [Selenium](https://www.selenium.dev/), for the given time period, for the specified weather station. This creates a Chrome web driver that automatically visits pages and extracts data into a csv file.

Steps to run:

1.  Set environment variables in `data.env`

        WEATHER_STATION="154210" # Bucuresti Otopeni
        START_YEAR="2017"
        END_YEAR="2022"

2.  Run the script

        (venv) $ python get_data.py

### Description

Depending on the time period specified by the user, the duration of the script may vary. After finishing, the created csv file will have the following columns:

- Year
- Month
- Day
- Temperature_Avg
- Temperature_Max
- Temperature_Min
- Sea_Level_Pressure
- Humidity_Avg
- Precipitation_Total
- Visibility_Avg
- Wind_Avg
- Wind_Sustained_Max
- Wind_Max
- Is_Rain
- Is_Snow
- Is_Storm
- Is_Fog

Example of the csv file created:

    Year,Month,Day,Temperature_Avg,Temperature_Max,Temperature_Min,Sea_Level_Pressure,Humidity_Avg,Precipitation_Total,Visibility_Avg,Wind_Avg,Wind_Sustained_Max,Wind_Max,Is_Rain,Is_Snow,Is_Storm,Is_Fog
    2017,01,1,-2,5,-6,,76,0,6,12.8,18.3,25.2,False,False,False,False
    2017,01,2,-1.6,6.3,-6,,71,0,7.2,12.6,24.1,28.7,False,False,False,False
    2017,01,3,-1.3,5.2,-7,,71,0,6.9,7.2,18.3,21.7,False,False,False,False
    2017,01,4,-2,4.6,-7.3,,75,0,6,11.3,22.2,25.2,False,False,False,False
