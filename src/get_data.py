import csv
import os
import re

from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

BASE_PATH = os.path.join("/", *os.environ["VIRTUAL_ENV"].split("/")[:-1])

# Load environment configuration
load_dotenv(os.path.join(BASE_PATH, "env", "data.env"))

# Setup webdriver
options = Options()
options.add_argument("--headless=new")
driver = webdriver.Chrome(options=options)

# Setup script variables from .env
weather_station = os.getenv("WEATHER_STATION")
start_year = int(os.getenv("START_YEAR"))
end_year = int(os.getenv("END_YEAR"))
years = [str(year) for year in range(start_year, end_year + 1)]

# Other script variables
months = [str(month).zfill(2) for month in range(1, 13)]
strong_re = r"<strong>(\d+)</strong>"
destination_path = os.path.join(
    "datasets", f"{weather_station}_{start_year}_{end_year}.csv"
)
try:
    os.makedirs("datasets")
except OSError:
    # Already exists, do nothing
    pass

measured_data = [
    "Year",
    "Month",
    "Day",
    "Temperature_Avg",
    "Temperature_Max",
    "Temperature_Min",
    "Sea_Level_Pressure",
    "Humidity_Avg",
    "Precipitation_Total",
    "Visibility_Avg",
    "Wind_Avg",
    "Wind_Sustained_Max",
    "Wind_Max",
    "Is_Rain",
    "Is_Snow",
    "Is_Storm",
    "Is_Fog",
]

table_xpath = "//*[@id='ColumnaIzquierda']/div/div[4]/table"

# Open csv file and write data according to table
with open(destination_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(measured_data)

    for year in years:
        for month in months:
            url = f"https://en.tutiempo.net/climate/{month}-{year}/ws-{weather_station}.html"

            driver.get(url)
            table = driver.find_element(By.XPATH, table_xpath)

            # Only interested in the rows that contain data, not the header
            rows = table.find_elements(By.TAG_NAME, "tr")[1:-2]

            for row in rows:
                tds = row.find_elements(By.TAG_NAME, "td")
                if row.find_elements(By.TAG_NAME, "span"):
                    # Badly formatted row, need to parse
                    data = [tds[0].get_attribute("innerHTML")]
                    # Last 4 contain categorical data, no need to look at
                    for td in tds[1:-4]:
                        # The actual value is represented inside css class
                        # attributes (˚ ˃̣̣̥⌓˂̣̣̥ )

                        spans = td.find_elements(By.TAG_NAME, "span")
                        content = ""
                        for span in spans:
                            number = driver.execute_script(
                                "var styles = window.getComputedStyle(arguments[0], '::after').content;return styles",
                                span,
                            )[1]
                            content += number
                        data.append(content)
                    for td in tds[-4:]:
                        data.append(td.get_attribute("innerHTML"))
                else:
                    # Well formatted row, can just read innerHTML for data
                    data = [v.get_attribute("innerHTML") for v in tds]

                # Process categorical rows
                data = [
                    el == "o"
                    if el in ["&nbsp;", "o"]
                    else ""
                    if el == "-"
                    else el
                    for el in data
                ]

                try:
                    data[0] = re.search(strong_re, data[0]).group(1)
                except AttributeError:
                    continue

                # Write the parsed data in the csv file
                arr = [year, month]
                arr.extend(data)
                writer.writerow(arr)
    driver.close()
