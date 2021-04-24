# Professional Academy UCD- Data analysis and Visualization on real- world Dataset

# 1) Real World Scenario
# Country Covid Confirmed cases data imported
# from https://www.kaggle.com/harikrishna9/covid19-dataset-by-john-hopkins-university as CSV file
# Country GDP data imported from https://www.kaggle.com/londeen/world-happiness-report-2020 as CSV file


# 2) Importing Data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

covid_dataset = pd.read_csv("RAW_global_confirmed_cases.csv")
print(covid_dataset.head())
print(covid_dataset.shape)
print(covid_dataset.info())

country_gdp = pd.read_csv("WHR20_DataForFigure2.1.csv")
print(country_gdp.head())
print(country_gdp.shape)
print(country_gdp.info())

# 3) Analyzing data

# Drop ‘Province/State’, Lat’ & ‘Long’ columns
covid_dataset.drop(['Province/State', 'Lat', 'Long'], axis=1, inplace=True)
print(covid_dataset.head())



# Group by - sum of people infected per country
covid_country = covid_dataset.groupby(covid_dataset.columns[0])[covid_dataset.columns[-1]].sum()
print("after sum")
print(covid_country.head())


covid_country_rename = covid_country.rename({'Country/Region ': 'Country name'}, axis=1, inplace=False)
print("covid_country_rename")
print(covid_country_rename.head())



# Slicing of Country GDP Data to exclude incomplete 2020  data
# gdp_data = country_gdp.loc[:, "Ladder score": "2020"]

# Subsetting country_gdp data to focus on logged GDP per capita
twentytwenty_gdp = country_gdp[["Country name", "Regional indicator", "Logged GDP per capita"]]
print(twentytwenty_gdp.head())



# Merging of GDP(GDP per capita per country)data with Covid infections per country
covid_gdp = country_gdp.merge(covid_country, on="Country name")
print("covid_gdp")
print(covid_gdp.head())

country_gdp = country_gdp.columns.str.upper
print(country_gdp)

# Create a list of EU Countries
EU_Countries = {"Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus","Denmark",
                "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy",
                "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal",
                "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"}

# Create a list of EU countries with similar populations
Select_EU_Countries = {"Finland", "Ireland", "Slovakia"}

print("covid_country.loc[Select_EU_Countries].head()")
print(covid_country.loc[Select_EU_Countries].head())

x = [Select_EU_Countries]
y = [10.4, 10.6,10.2]
plt.show()

# 5) Visualize Data
# Visualize the data for Finland, Ireland & Slovakia
covid_country.loc[Select_EU_Countries].plot(kind='bar', x='GDP(%)', y='EU Countries', color='red')
plt.show()
plt.title("Covid Infections and GDP")
plt.ylabel("Covid Cases")
plt.xlabel("EU Countries")

country_gdp["Logged GDP per capita"] .value_counts()
country_gdp.plot(kind="scatter" , x="Country name", y="Logged GDP per capita")
plt.show()

# Compare the number of people infected with covid and GDP per capita
# in 3 European Countries with similar populations Finland, Ireland & Slovakia
# sns.set_theme(style="whitegrid")
# sns.barplot(y='Country Name', x='Number_of_Pop_infected', data=covid_country.loc[Select_EU_Countries], color="blue")
# sns.barplot(y='Country Name', x='Logged GDP per capita', data=covid_country, color="blue")
# plt.bar(x_array, y_array, color='green')
# plt.title("Covid Infections and GDP")
# plt.xlabel("GDP(%)")
# plt.ylabel("EU Countries")
# plt.show()

import matplotlib.pyplot as plt
import csv

Names = []
Values = []

with open('WHR20_DataForFigure2.4.csv', 'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        Names.append(row[0])
        Values.append(str(row[6]))

plt.scatter(Values, Names, color='g', s=100)
plt.xticks(rotation=25)
plt.xlabel('Names')
plt.ylabel('Values')
plt.title('Logged GDP per Capita', fontsize=20)

plt.show()
# 6) Data Insights
# 1	Ireland has 3 times the number of people infected compared to Finland
# 2	Slovakia has 4.5 times the number of people infected compared to Finland
# 3	Slovakia has the lowest GPD per capita of the 3 countries and the highest number of people infected with Covid
# 4	Finland has the 2nd highest GDP per capita amongst these 3 countries but the lowest Covid infection rates
# 5 There is a correlation between GDP per capita and life expectancy, Ireland has the highest Life Expectancy and the highest GDP per capita.