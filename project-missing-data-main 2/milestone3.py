from datetime import datetime

from datetime import date

import requests

import json

import airflow

from airflow import DAG

from airflow.operators.python_operator import PythonOperator

from airflow.operators.docker_operator import DockerOperator


# extracting the data for the first two data sets athlete_events and noc_regions
def extract_data():
    import pandas as pd

    df_olympics_raw = pd.read_csv('./data/athlete_events.csv')

    df_regions_raw = pd.read_csv('./data/noc_regions.csv')

    print(df_olympics_raw.info())

    print(df_regions_raw.info())

    # saving to transformations file
    df_olympics_raw.to_csv('./transformations/athletes_df_raw.csv', index=False)

    df_regions_raw.to_csv('./transformations/regions_df_raw.csv', index=False)


# removing duplicate rows, handling missing words and fixing the spelling for some words
def prepare_athletes_dataset():
    import pandas as pd

    import numpy as np

    df_olympics = pd.read_csv('./transformations/athletes_df_raw.csv')

    # Handle Missing Words

    df_olympics.replace('NA', np.nan, inplace=True)

    df_olympics.replace('Missing', np.nan, inplace=True)

    # Remove Duplicate Rows

    duplicateData = df_olympics[df_olympics.duplicated(keep='last')].sort_values(by=['Name'], ascending=True)

    duplicateIndexes = duplicateData.index

    df_olympics = df_olympics.drop(duplicateIndexes)

    # Fixing Spelling

    df_olympics['City'].replace('Athina', 'Athens', inplace=True)

    df_olympics['City'].replace('Moskva', 'Moscow', inplace=True)

    # saving to transformations file
    df_olympics.to_csv('./transformations/athletes_df_prepared.csv', index=False)


# renaming countries and regions to a suitable name
def prepare_regions_dataset():
    import pandas as pd

    import numpy as np

    df_regions = pd.read_csv('./transformations/regions_df_raw.csv')

    df_regions['region'].replace('SIN', 'SGP', inplace=True)

    df_regions.drop(['notes'], axis=1, inplace=True)

    df_regions['region'].loc[(df_regions['NOC'] == 'ROT')] = 'Refugee Olympic Team'

    df_regions['region'].loc[(df_regions['NOC'] == 'TUV')] = 'Tuvalu'

    for index in df_regions.index:

        if df_regions.loc[index, 'region'] == 'Montenegro':

            df_regions.loc[index, 'region'] = 'Serbia'

        elif df_regions.loc[index, 'region'] == 'Curacao':

            df_regions.loc[index, 'region'] = 'Netherlands'

        elif df_regions.loc[index, 'region'] == 'Kosovo':

            df_regions.loc[index, 'region'] = 'Serbia'

    df_regions.dropna(axis='index', subset=['region'], inplace=True)
    # saving to transformations file
    df_regions.to_csv('./transformations/regions_df_prepared.csv', index=False)


# here we handle age, weight and height outliers each in a separate function
def handle_outliers():
    import pandas as pd

    df = pd.read_csv('./transformations/athletes_df_prepared.csv')

    print(df.describe())

    df1 = handle_age_outliers(df)

    df2 = handle_weight_outliers(df1)

    df3 = handle_height_outliers(df2)
    # final data frame which is saved
    print(df3.describe())
    # saving to transformations file
    df3.to_csv('./transformations/athletes_no_outliers.csv', index=False)


# here we handle age, weight and height missing values each in a separate function
def impute_missing_data():
    import pandas as pd

    df = pd.read_csv('./transformations/athletes_no_outliers.csv')

    print(df.isna().sum())

    df1 = handle_missing_age(df)

    df2 = handle_missing_weight(df1)

    df3 = handle_missing_height(df2)
    # final data frame which is saved
    print(df3.isna().sum())
    # saving to transformations file
    df3.to_csv('./transformations/athletes_no_missing.csv', index=False)


# in this function we do a label encoding to the medals
def enumerate_medals():
    import pandas as pd

    import numpy as np

    df_olympics = pd.read_csv('./transformations/athletes_no_missing.csv')

    medals = []  # define array structure

    for medal in df_olympics["Medal"]:

        if medal == 'Gold':

            medals.append(3)

        elif medal == 'Silver':

            medals.append(2)

        elif medal == 'Bronze':

            medals.append(1)

        else:

            medals.append(0)

    # Copy dataframe to keep original

    df_olympics["Medal_Enumarated"] = medals

    del df_olympics["Medal"]
    # saving to transformations file
    df_olympics.to_csv('./transformations/athletes_encoded.csv', index=False)


# merging the two data sets df_olympics and df_regions after data cleaning
def merge_athletes_regions():
    import pandas as pd

    import numpy as np

    df_olympics_latest = pd.read_csv('./transformations/athletes_encoded.csv')

    df_regions_latest = pd.read_csv('./transformations/regions_df_prepared.csv')

    df_athletes_merged = pd.merge(df_olympics_latest, df_regions_latest, left_on='NOC', right_on='NOC')

    df_athletes_merged = df_athletes_merged[df_athletes_merged['region'] != 'Individual Olympic Athletes']

    df_athletes_merged = df_athletes_merged[df_athletes_merged['region'] != 'Refugee Olympic Team']

    df_athletes_merged = df_athletes_merged[df_athletes_merged['region'] != 'Israel']  # Because it is not a country

    # saving to transformations file
    df_athletes_merged.to_csv('./transformations/olympics_regions_merged.csv', index=False)


# extracting the countries of the world.csv data set and cleaning it and merging it to our data set
def merge_athletes_countries():
    import pandas as pd

    df_olympics = pd.read_csv('./transformations/olympics_regions_merged.csv')

    df_olympics_countries = all_functions_for_merging(df_olympics)

    print(df_olympics_countries)
    # saving to transformations file
    df_olympics_countries.to_csv('./transformations/athletes_countries_merged.csv', index=False)

# adds a BMI feature to the data frame 
def add_BMI_feature():
    import pandas as pd

    df_olympics = pd.read_csv('./transformations/athletes_countries_merged.csv')

    df_olympics['BMI'] = df_olympics['Weight'] / (df_olympics['Height'] / 100) ** 2

    df_olympics.to_csv('./transformations/BMI_feature.csv', index=False)

# adds a Hosting Country feature to the data frame 
def add_Hosting_Country_feature():
    import pandas as pd

    df_olympics = pd.read_csv('./transformations/BMI_feature.csv')

    df_olympics_hosting = add_hosting_country(df_olympics)

    df_olympics_hosting.to_csv('./transformations/Hosting_Feature.csv', index=False)
# save the data into a final csv
def save_data():
    import pandas as pd

    df_olympics = pd.read_csv('./transformations/Hosting_Feature.csv')

    df_olympics.to_csv('./transformations/final_dataset.csv', index=False)


# retireive the mean age without outliers
def get_age_mean_without_outliers(sport_df):
    import pandas as pd

    import numpy as np

    Q1 = sport_df.Age.quantile(0.25)

    Q2 = sport_df.Age.quantile(0.5)

    Q3 = sport_df.Age.quantile(0.75)

    IQR = Q3 - Q1

    left_whisker = Q1 - 1.5 * IQR

    right_whisker = Q3 + 1.5 * IQR

    sport_inliers = sport_df[(sport_df.Age <= right_whisker) & (sport_df.Age >= left_whisker)]

    mean = sport_inliers.Age.mean()

    return {

        'right_whisker': right_whisker,

        'left_whisker': left_whisker,

        'mean': int(mean)

    }

# remove outliers from age 
def handle_age_outliers(athletes_df):
    import pandas as pd

    import numpy as np

    athletes_df_no_age_outliers = pd.DataFrame()

    n_outliers = 0

    olympics_sports = athletes_df.Sport.unique()

    for s in olympics_sports:
        sport_df = athletes_df[athletes_df.Sport == s]

        outliers_dict = get_age_mean_without_outliers(sport_df=sport_df)

        outliers_age_list = np.array(sport_df.loc[(sport_df.Age < outliers_dict['left_whisker']) | (
                    sport_df.Age > outliers_dict['right_whisker']), 'Age'])

        n_outliers += len(outliers_age_list)

        sport_df.Age.replace(outliers_age_list, outliers_dict['mean'], inplace=True)

        athletes_df_no_age_outliers = athletes_df_no_age_outliers.append(sport_df)

    return athletes_df_no_age_outliers

# remove outliers from wieght
def handle_weight_outliers(df):
    import pandas as pd

    df_no_weight_outliers_F = RemovingWeightOutliersPerGender(df, 'F')

    df_no_weight_outliers_FM = RemovingWeightOutliersPerGender(df_no_weight_outliers_F, 'M')

    return df_no_weight_outliers_FM

# remove outliers from height
def handle_height_outliers(df):
    import pandas as pd

    df_no_height_outliers_F = RemovingWeightOutliersPerGender(df, 'F')

    df_no_height_outliers_FM = RemovingWeightOutliersPerGender(df_no_height_outliers_F, 'M')

    return df_no_height_outliers_FM

# handle wieght outliers for each gender seperatly 
def RemovingWeightOutliersPerGender(df_athletes, gender):
    import numpy as np

    import pandas as pd

    df_olympics = df_athletes.copy()

    Events = df_olympics["Event"].unique()

    eventsWithEmptyWeight = np.array([])

    for event in Events:

        temp = df_olympics[(df_olympics.Sex == gender) & (df_olympics.Event == event)]

        if temp.shape[0] == temp['Weight'].isna().sum():
            eventsWithEmptyWeight = np.append(eventsWithEmptyWeight, [event])

    validEvents = df_olympics['Event'].unique()

    validEvents = np.setdiff1d(validEvents, eventsWithEmptyWeight)

    for event in validEvents:
        Q1 = df_olympics["Weight"].quantile(0.25)

        Q3 = df_olympics["Weight"].quantile(0.75)

        IQR = Q3 - Q1

        cut_off = IQR * 1.5

        lower = Q1 - cut_off

        upper = Q3 + cut_off

        df = df_olympics[
            (df_olympics["Weight"] >= lower) & (df_olympics["Weight"] <= upper) & (df_olympics.Sex == gender) & (
                        df_olympics.Event == event)]

        mean = df["Weight"].mean()

        df_olympics["Weight"].mask(
            ((df_olympics["Weight"] < lower) | (df_olympics["Weight"] > upper)) & (df_olympics.Sex == gender) & (
                        df_olympics.Event == event), mean, inplace=True)

    return df_olympics

# handle hieght outliers for each gender seperatly 
def RemovingHeightOutliersPerGender(df_athletes, gender):
    import numpy as np

    import pandas as pd

    df_olympics = df_athletes.copy()

    Sports = df_olympics["Sport"].unique()

    sportsWithEmptyHeight = np.array([])

    for sport in Sports:

        temp = df_olympics[(df_olympics.Sex == gender) & (df_olympics.Sport == sport)]

        if temp.shape[0] == temp['Height'].isna().sum():
            sportsWithEmptyHeight = np.append(sportsWithEmptyHeight, [sport])

    validSports = df_olympics['Sport'].unique()

    validSports = np.setdiff1d(validSports, sportsWithEmptyHeight)

    for sport in validSports:
        Q1 = df_olympics["Height"].quantile(0.25)

        Q3 = df_olympics["Height"].quantile(0.75)

        IQR = Q3 - Q1

        cut_off = IQR * 1.5

        lower = Q1 - cut_off

        upper = Q3 + cut_off

        df = df_olympics[
            (df_olympics["Height"] >= lower) & (df_olympics["Height"] <= upper) & (df_olympics.Sex == gender) & (
                        df_olympics.Sport == sport)]

        mean = df["Height"].mean()

        df_olympics["Height"].mask(
            ((df_olympics["Height"] < lower) | (df_olympics["Height"] > upper)) & (df_olympics.Sex == gender) & (
                        df_olympics.Sport == sport), mean, inplace=True)

    return df_olympics

# impute the missing age
def handle_missing_age(df):
    import pandas as pd

    import numpy as np

    df_olympics = df.copy()

    avgAges = df_olympics.groupby(['Sport'])["Age"].mean()

    df_olympics['Age'].fillna(df_olympics['Sport'].map(avgAges), inplace=True)

    return df_olympics

# impute the missing weight
def handle_missing_weight(df):
    import pandas as pd

    import numpy as np

    df_olympics = df.copy()

    malesData = df_olympics[df_olympics['Sex'] == 'M']

    femalesData = df_olympics[df_olympics['Sex'] == 'F']

    averageMaleWeightPerEvent = malesData.groupby(['Event'])["Weight"].mean()

    malesData['Weight'].fillna(malesData['Event'].map(averageMaleWeightPerEvent), inplace=True)

    averageFemaleWeightPerEvent = femalesData.groupby(['Event'])["Weight"].mean()

    femalesData['Weight'].fillna(femalesData['Event'].map(averageFemaleWeightPerEvent), inplace=True)

    averageMaleWeightPerSport = malesData.groupby(['Sport'])["Weight"].mean()

    malesData['Weight'].fillna(malesData['Sport'].map(averageMaleWeightPerSport), inplace=True)

    averageFemaleWeightPerSport = femalesData.groupby(['Sport'])["Weight"].mean()

    femalesData['Weight'].fillna(femalesData['Sport'].map(averageFemaleWeightPerSport), inplace=True)

    frames = [malesData, femalesData]

    df_olympics = pd.concat(frames)

    averageWeightPerGender = df_olympics.groupby(['Sex'])["Weight"].mean()

    df_olympics['Weight'].fillna(df_olympics['Sex'].map(averageWeightPerGender), inplace=True)

    return df_olympics

# impute the missing height
def handle_missing_height(df):
    import pandas as pd

    import numpy as np

    df_olympics = df.copy()

    # 1- Mean by gender and sport

    # 2- Mean by gender

    malesData = df_olympics[df_olympics['Sex'] == 'M']

    femalesData = df_olympics[df_olympics['Sex'] == 'F']

    averageMaleHeightPerSport = malesData.groupby(['Sport'])["Height"].mean()

    malesData['Height'].fillna(malesData['Sport'].map(averageMaleHeightPerSport), inplace=True)

    averageFemaleHeightPerSport = femalesData.groupby(['Sport'])["Height"].mean()

    femalesData['Height'].fillna(femalesData['Sport'].map(averageFemaleHeightPerSport), inplace=True)

    frames = [malesData, femalesData]

    df_olympics = pd.concat(frames)

    averageHeightPerGender = df_olympics.groupby(['Sex'])["Height"].mean()

    df_olympics['Height'].fillna(df_olympics['Sex'].map(averageHeightPerGender), inplace=True)

    return df_olympics

# extract countries data set
def extract_countries():
    import pandas as pd

    df_countries = pd.read_csv('./data/countries of the world.csv')

    return df_countries

# delete unneeded columns
def delete_columns(df_countries):
    import numpy as np

    import pandas as pd

    df_countries = df_countries.copy()

    cols = df_countries.columns.values

    removed_cols = np.delete(cols, np.where(cols == 'Country'))

    removed_cols = np.delete(removed_cols, np.where(removed_cols == 'GDP ($ per capita)'))

    # removed_cols = np.delete(removed_cols, np.where(removed_cols == 'Population'))

    print(removed_cols)

    df_countries.drop(removed_cols, axis=1, inplace=True)

    return df_countries

# get an array of all unique countries
def get_countries(df_olympics):
    import pandas as pd

    countries = df_olympics['region'].unique()

    countries.sort()

    return countries

# remove extra space in the name of some countries
def remove_extra_space(df_countries, countries):
    import numpy as np

    import pandas as pd

    from textdistance import levenshtein

    df_countries = df_countries.copy()

    countries_external = df_countries['Country']

    for i in range(len(countries_external)):
        countries_external[i] = countries_external[i][:-1]

    countries_external

    df_countries_separated = df_countries[
        (df_countries['Country'] == 'Sudan') | (df_countries['Country'] == 'Virgin Islands')]

    print(df_countries_separated)

    extra1 = {'Country': 'South Sudan', 'GDP ($ per capita)': '1900.0'}

    extra2 = {'Country': 'Virgin Islands, US', 'GDP ($ per capita)': '17200.0'}

    df_countries.loc[220, 'Country'] = 'Virgin Islands, British'

    df_countries = df_countries.append(extra1, ignore_index=True)

    df_countries = df_countries.append(extra2, ignore_index=True)

    df_countries[(df_countries['Country'] == 'Sudan') | (df_countries['Country'] == 'Virgin Islands, British') | (
                df_countries['Country'] == 'South Sudan') | (df_countries['Country'] == 'Virgin Islands, US')]

    lev_matrix = np.zeros((len(countries), len(countries_external)))

    for i in range(lev_matrix.shape[0]):

        for j in range(lev_matrix.shape[1]):
            lev_matrix[i][j] = levenshtein.distance(countries[i], countries_external[j])

    print(lev_matrix)

    dict_replacement = {}

    for i in range(lev_matrix.shape[0]):

        min = 100000

        minidx = -1

        for j in range(lev_matrix.shape[1]):

            if lev_matrix[i][j] < min:
                min = lev_matrix[i][j]

                minidx = j

        if countries_external[minidx] != countries[i]:
            dict_replacement[countries_external[minidx]] = countries[i]

    print(dict_replacement)

    return df_countries

# rename some countries to the correct name
def renaming(df):
    df2 = df.copy()

    for index in df2.index:
        if df2.loc[index, 'Country'] == 'Antigua & Barbuda':

            df2.loc[index, 'Country'] = 'Antigua'

        elif df2.loc[index, 'Country'] == 'Bahamas, The':

            df2.loc[index, 'Country'] = 'Bahamas'

        elif df2.loc[index, 'Country'] == 'Congo, Repub. of the':

            df2.loc[index, 'Country'] = 'Republic of Congo'

        elif df2.loc[index, 'Country'] == 'Congo, Dem. Rep.':

            df2.loc[index, 'Country'] = 'Democratic Republic of the Congo'

        elif df2.loc[index, 'Country'] == "Cote d'Ivoire":

            df2.loc[index, 'Country'] = 'Ivory Coast'

        elif df2.loc[index, 'Country'] == 'Gambia, The':

            df2.loc[index, 'Country'] = 'Gambia'

        elif df2.loc[index, 'Country'] == 'United States':

            df2.loc[index, 'Country'] = 'USA'

        elif df2.loc[index, 'Country'] == 'Israel':

            df2.loc[index, 'Country'] = 'Palestine'

        elif df2.loc[index, 'Country'] == 'East Timor':

            df2.loc[index, 'Country'] = 'Timor-Leste'

        elif df2.loc[index, 'Country'] == 'Burma':

            df2.loc[index, 'Country'] = 'Myanmar'

        elif df2.loc[index, 'Country'] == 'Saint Kitts & Nevis':

            df2.loc[index, 'Country'] = 'Saint Kitts'

        elif df2.loc[index, 'Country'] == 'Saint Vincent and the Grenadines':

            df2.loc[index, 'Country'] = 'Saint Vincent'

        elif df2.loc[index, 'Country'] == 'Trinidad & Tobago':

            df2.loc[index, 'Country'] = 'Trinidad'

        elif df2.loc[index, 'Country'] == 'Micronesia, Fed. St.':

            df2.loc[index, 'Country'] = 'Micronesia'

        elif df2.loc[index, 'Country'] == 'Burma':

            df2.loc[index, 'Country'] = 'Myanmar'

        elif df2.loc[index, 'Country'] == 'United Kingdom':

            df2.loc[index, 'Country'] = 'UK'

        elif df2.loc[index, 'Country'] == 'Korea, South':

            df2.loc[index, 'Country'] = 'South Korea'

        elif df2.loc[index, 'Country'] == 'Korea, North':

            df2.loc[index, 'Country'] = 'North Korea'

        elif df2.loc[index, 'Country'] == 'Central African Rep.':

            df2.loc[index, 'Country'] = 'Central African Republic'

        elif df2.loc[index, 'Country'] == 'Bosnia & Herzegovina':

            df2.loc[index, 'Country'] = 'Bosnia and Herzegovina'

        elif df2.loc[index, 'Country'] == 'Bolivia':

            df2.loc[index, 'Country'] = 'Boliva'

        elif df2.loc[index, 'Country'] == 'Sao Tome & Principe':

            df2.loc[index, 'Country'] = 'Sao Tome and Principe'

        elif df2.loc[index, 'Country'] == 'Virgin Islands':

            df2.loc[index, 'Country'] = 'Virgin Islands, US'

    new_countries = df2['Country'].unique()

    new_countries.sort()

    print(new_countries.__contains__('Antigua & Barbuda'))

    print(new_countries.__contains__('Bahamas, The'))

    return df2

# change the column country with the name region
def change_country_region(df_countries):
    df = df_countries.copy()

    df.rename(columns={'Country': 'region'}, inplace=True)

    return df
# merge olypics data set with countries data set

def merge_at_last(df_olympics, df_countries):
    import pandas as pd

    df_merged_countries = pd.merge(df_olympics, df_countries, how='left', on='region')

    return df_merged_countries


# merging the countries of the world.csv and the df_olympics after cleaning both
def all_functions_for_merging(df_olympics):
    df_countries = extract_countries()

    df_countries = delete_columns(df_countries)

    countries = get_countries(df_olympics)

    df_countries = remove_extra_space(df_countries, countries)

    df_countries = renaming(df_countries)

    changed_df_countries = change_country_region(df_countries)
    df_merged_countries = merge_at_last(df_olympics, changed_df_countries)

    return df_merged_countries

# hosting country coresponding to each city
def add_hosting_country(df):
    import pandas as pd

    german_cities = ['Berlin', 'Munich', 'Garmisch-Partenkirchen']

    italian_cities = ['Roma', 'Torino', "Cortina d'Ampezzo"]

    spanish_cities = ['Barcelona']

    canadian_cities = ['Montreal', 'Vancouver', 'Calgary']

    american_cities = ['Atlanta', 'Los Angeles', 'St. Louis', 'Lake Placid', 'Salt Lake City', 'Squaw Valley']

    french_cities = ['Paris', 'Albertville', 'Grenoble', 'Chamonix']

    chineese_cities = ['Beijing']

    greek_cities = ['Athens']

    british_cities = ['London']

    brasilian_cities = ['Rio de Janeiro']

    korean_cities = ['Seoul']

    australian_cities = ['Sydney', 'Melbourne']

    japanese_cities = ['Tokyo', 'Nagano', 'Sapporo']

    mexican_cities = ['Mexico City']

    finnish_cities = ['Helsinki']

    russian_cities = ['Moscow', 'Sochi']

    belgian_cities = ['Antwerpen']

    dutch_cities = ['Amsterdam']

    swedish_cities = ['Stockholm']

    bosnian_cities = ['Sarajevo']

    norwegian_cities = ['Lillehammer', 'Oslo']

    swiss_cities = ['Sankt Moritz']

    austrian_cities = ['Innsbruck']

    df2 = df.copy()

    df2['Hosting Country'] = pd.Series()

    df2['Hosting Country'].loc[df['City'].isin(german_cities)] = 'Germany'

    df2['Hosting Country'].loc[df['City'].isin(italian_cities)] = 'Italy'

    df2['Hosting Country'].loc[df['City'].isin(spanish_cities)] = 'Spain'

    df2['Hosting Country'].loc[df['City'].isin(canadian_cities)] = 'Canada'

    df2['Hosting Country'].loc[df['City'].isin(american_cities)] = 'USA'

    df2['Hosting Country'].loc[df['City'].isin(french_cities)] = 'France'

    df2['Hosting Country'].loc[df['City'].isin(chineese_cities)] = 'China'

    df2['Hosting Country'].loc[df['City'].isin(greek_cities)] = 'Greece'

    df2['Hosting Country'].loc[df['City'].isin(british_cities)] = 'UK'

    df2['Hosting Country'].loc[df['City'].isin(brasilian_cities)] = 'Brazil'

    df2['Hosting Country'].loc[df['City'].isin(korean_cities)] = 'South Korea'

    df2['Hosting Country'].loc[df['City'].isin(australian_cities)] = 'Australia'

    df2['Hosting Country'].loc[df['City'].isin(japanese_cities)] = 'Japan'

    df2['Hosting Country'].loc[df['City'].isin(mexican_cities)] = 'Mexico'

    df2['Hosting Country'].loc[df['City'].isin(finnish_cities)] = 'Finland'

    df2['Hosting Country'].loc[df['City'].isin(russian_cities)] = 'Russia'

    df2['Hosting Country'].loc[df['City'].isin(dutch_cities)] = 'Netherlands'

    df2['Hosting Country'].loc[df['City'].isin(chineese_cities)] = 'China'

    df2['Hosting Country'].loc[df['City'].isin(belgian_cities)] = 'Belgium'

    df2['Hosting Country'].loc[df['City'].isin(swedish_cities)] = 'Sweden'

    df2['Hosting Country'].loc[df['City'].isin(bosnian_cities)] = 'Bosnia and Herzegovina'

    df2['Hosting Country'].loc[df['City'].isin(norwegian_cities)] = 'Norway'

    df2['Hosting Country'].loc[df['City'].isin(swiss_cities)] = 'Switzerland'

    df2['Hosting Country'].loc[df['City'].isin(austrian_cities)] = 'Austria'

    return df2


default_args = {

    'owner': 'airflow',

    'depends_on_past': False,

    'start_date': datetime(2021, 12, 20),

    'catchup': False

}

# dag
dag = DAG(

    'olympics_dag',

    default_args=default_args,

    description='A DAG to extract, transform, load olympics data over 120 years.',

    schedule_interval='@once'

)

extract_data_task = PythonOperator(

    task_id='extract_data',

    python_callable=extract_data,

    provide_context=True,

    dag=dag

)

preparing_athletes_task = PythonOperator(

    task_id='prepare_athletes_dataset',

    python_callable=prepare_athletes_dataset,

    provide_context=True,

    dag=dag

)

preparing_regions_task = PythonOperator(

    task_id='prepare_regions_dataset',

    python_callable=prepare_regions_dataset,

    provide_context=True,

    dag=dag

)

outliers_task = PythonOperator(

    task_id='handle_outliers',

    python_callable=handle_outliers,

    provide_context=True,

    dag=dag

)

missing_data_task = PythonOperator(

    task_id='impute_missing_data',

    python_callable=impute_missing_data,

    provide_context=True,

    dag=dag

)

merge_regions_task = PythonOperator(

    task_id='merge_athletes_regions',

    python_callable=merge_athletes_regions,

    provide_context=True,

    dag=dag

)

merge_countries_task = PythonOperator(

    task_id='merge_athletes_countries',

    python_callable=merge_athletes_countries,

    provide_context=True,

    dag=dag

)

enumerate_medals_task = PythonOperator(

    task_id='enumerate_medals',

    python_callable=enumerate_medals,

    provide_context=True,

    dag=dag

)

BMI_task = PythonOperator(

    task_id='BMI_feature',

    python_callable=add_BMI_feature,

    provide_context=True,

    dag=dag

)

Hosting_Country_task = PythonOperator(

    task_id='Hosting_Country',

    python_callable=add_Hosting_Country_feature,

    provide_context=True,

    dag=dag

)


Save_data_task = PythonOperator(

    task_id='save_data',

    python_callable=save_data,

    provide_context=True,

    dag=dag

)



# Each of our nodes calls a function that calls other functions which are implemented in milestone 1 or milestone 2.
# Our node functions are at the start of the file


# dependencies
extract_data_task >> preparing_athletes_task >> preparing_regions_task >> outliers_task >> missing_data_task >> enumerate_medals_task >> merge_regions_task >> merge_countries_task>>  BMI_task >> Hosting_Country_task >> Save_data_task

# merge_countries_task