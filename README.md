 
# Data Engineering-Project
Our main goal is to observe the factors that affect the performance. By performance, we mean winning medals, which also includes different ranks of the medals.

## Research questions
1. Does economy of the country affect their performance? (merging 1)
2. What is the estimated number of appearances for a participant to achieve their first medal?
3. What are the features that most affect player performance?
4. Predict whether the player could get a medal or not next year?
5. How different BMI ranges are demanded, and what are the effects and changes of BMI? (feature engineering 1)
6. How hosting events affects country's performance? (feature engineering 2)
7. Do countries with high gdp are chosen to host the olympics ?
8. Do the number of medals on average increase as an athlete participates more?
9. What are the performances of both genders?
10. Is the number of medals related to the number of each medal?

## The datasets
- At the beginning we read the two datasets, athletes and regions. As we were exploring the data, we considered joining the athlete dataset with the regions dataset on the NOC attribute which was found in both datasets. Yet, we found out that the number of data objects were reduced after merging the two datasets. We then figured out that this behaviour was a result of several objects with the attribute NOC equal to SGP that did not appear in regions dataset. But also we had a SIN attribute that appeared in regions but not athletes dataset. Where both were actually representing the same country, hence we replaced SGP regions with SIN and merged data.


## EDA / Visualization before cleaning the data
- We then explored the columns, their types and some other information. 
- Our dataset includes 17 columns, 5 of which are numeric, and 270767 entries.
- We found 1385 items that were duplicated, that is the attributes of the entire column is duplicated, which we dropped.
- Age distribution
- Weight distribution
- Height distribution
- Box plots for weight and height ber gender --> to see overall skewness --> we later found that the range of age is different by the events
- Bar chart showing frequency of players per season over the years --> we found that x values range were not uniform, after investigating we found that summer games are made each leap year (every 4 years) and winter games are made after two years from each leap year, yet winter games are named by the leap years --> we arranged the data so that years are leap years and seasons are either summer and winter.
- Others are described in the visualisation section.

## **Cleaning the data**
## Outliers
- For all outliers we plotted a graph showing outliers before and after imputing them.
- **Age** --> We handled age outliers with two functions, where the first function takes the dataframe for a single sport and calculates the right and left whiskers as well as the mean of the entries that are not considered outliers. Second function replaces the outliers of the age column for each sport with the mean of the age of participants without the outliers. In other words, replacing age outliers with the mean per sport.
- **Weight** --> We handled outliers first separating by gender for each event, but first by filtering the events that have no weights first then looping on the events, and using inter-quartile range to calculate the average of the values which aren't outliers, then assigning this value to the outliers. In other words, this means that w replaced outliers with the mean per event and gender. We choose weight specifically as some events had defined weight ranges for participants.
- **Height** --> Similar to the weight, we handled outliers first separating by gender for each sport, but first by filtering the events that have no weights first then looping on the sports, and using inter-quartile range to calculate the average of the values which aren't outliers, then assigning this value to the outliers. In other words, this means that w replaced outliers with the mean per sport and gender.

## Missing values
- We first unified all values that represented missing values to NaN 
- We then investigated the number of null values in each column.
- We also found that we had about 1385 duplicated items in our dataset.
- We found that the Age, Height, Weight, Medals, Regions and Notes had missing values. Hence we tried to fill the missing values in the most sensible way and we tried to consider the different types of missing values (MCAR, MAR, MNAR).
- We found that about 3% of our dataset had all three attributes missing, age, height and weight.
- We found around 229617 missing values in Medals column, which represented not winning any medal, hence we did not alter these values. (Encoded later)
- We figured out that three NOC had no corresponding regions
- Notes --> this column is not needed and full of NaNs, hence it was dropped. 
- Regions --> were filled according to their corresponding NOC and the unfigured or unknown regions were dropped --> 2 rows only so will not affect our data
- Age --> We imputed missing age values by the mean value per sport
- Weight --> We imputed missing weight values by the mean value per event and gender, because events participations have a standard weight. The remaining ones are handled by filtering over sport and gender. Then the remaining ones are handled by gender only.
- Height --> We imputed missing height values by the mean value per sport and gender. Then the remaining ones are done by gender only.
- We made sure after filling these columns that the missing values in all columns (except medla) is equal to zero.

## Inconsistencies
- Multiple NOCs belong to the same region --> Decided to work with regions as it is more accurate (NOC can be dropped later)
- Cities with multiple spellings --> Fixed spellings

## Encoding
- Encoded medal with label encoding, that is Gold, Silver, Bronze and no medal(NaN) became 3, 2, 1, and 0 respectively. Named this column as 'Medal_Enumarated'
- Deleted 'Medal' column as it is redundant by now

## Visualization
- Medals over the years for males
- Medals over the years for females
- Number of Medals won by USA in the years
- Scatterplot matrix showing correlation between the 3 quantitative attributes --> pairplot between age, height, and weight
- Number of Participants Per Gender
- Number of Medals Per Gender
- Total number of participations versus total number of medals per gender
- Age Histogram
- Age Histogram After filtering oldies (>50)
- Weight Distribution
- Height Distribution
- Box plot for age
- Frequency of players per season over the years
- Sports with age above 50
- Number of times each country hosted
- Germany's Performance as a Host Year vs other Years
- Top 10 countries winning medals in winter
- Top 10 countries winning medals in summer
- Top 10 countries winning medals overall
- Top Athletes winning medals in All Games
- Box plots for Age attribute for each sport
- Comparison of Total Medals Between USA and Russia
- Composition of medals between top 2 performers --> Each Medal for USA versus Russia
- Most Attending Players against their number of winning medals
- Players with most medals
- Composition of Medals of the top 10 regions
- Composition of Medals of the top 10 performers
- Total Medals per Gender
- Countries winning the most number of each medal --> Top countries with each medal (Gold, Bronze, Silver) 
- Number of events in each sport over years in summer
- Number of events in each sport over years in winter

## Merging datasets
- We merged our dataset with the countries dataset
- The countries dataset (https://www.kaggle.com/nitishabharathi/gdp-per-capita-all-countries)
- The countries dataset was explored and cleaned as well
- There was some data with missing values and some data with extra spaces that were trimmed
- There were some countries that either were divided into two now or two countries that were merged to a single country, this was handled too.
- We used levenstein distance for country names to resolve inconsistencies in naming to perform the merging of 2 datasets smoothly
- Both Individual Olympic Athletes and Refugee Olympic Team will be dropped as they do not belong to a certain country so they will not be of use in our analysis of countries
- We dropped Israel as it is not a country

## Feature Engineering
- We had two extra atrributes
- The BMI that was extracted from the weight and height in metric measurements --> We tried to find the demanded BMI and the correlation between the BMI and athlet's performance.
- The  isHosting were for each event we try to get the country hosted this event --> We tried to find if when a country hosts an event this increases its performance that is the number of medals won during these events. 

## Insights
### Olympics Seasons
We realised that the ranges of the x-labels were inconsistent. After investigating more we found out that olympics occurred every four years, that is in every leap year to be specific. We also found out that winter seasons started in 1994. The Winter and Summer olympics were held in the same years until 1992, where after that the winter games were held after two years from the summer games.

Researching more about the topic we found out that the International Olympic Committee (IOC) decided to place the Summer and Winter Games on separate four-year cycles in alternating even-numbered years.

Note that the gap between 1936 and 1948 was due to World War 2.

-> We found this when we were plotting the number of participants over the years by the season, were we found inconsistencies in the years.


### Rivalry between USA and Russia (top 2 performers)
We found that the top two countries from number of participants and number of medals were Russia and USA. We decided to observe the rivalry between these two countries, and we found out that although Russia had more participants over the years, but USA won more medals in almost all the years.

-> We tried observing the top two performers, which were dominant in our dataset, and hence we did a stacked bar chart for the comparison of total medals between USA and Russia, along with another stacked bard chart for each medal type for USA versus Russia.

## Explorations 
1. Does economy of the country affect their performance? (merging 1)
2. What is the estimated number of appearances for a participant to achieve their first medal?
3. What are the features that most affect player performance?
4. Predict whether the player could get a medal or not next year?
5. How different BMI ranges are demanded, and what are the effects and changes of BMI? (feature engineering 1)
6. How hosting events affects country's performance? (feature engineering 2)
7. Do countries with high gdp are chosen to host the olympics ?
8. Do the number of medals on average increase as an athlete participates more?
9. What are the performances of both genders?
10. Is the number of medals related to the number of each medal?

### 1. Does economy of the country affect their performance? (merging 1)
We can notice that there is a moderate correlation between GDP and total medals achieved because the correlation value in both pearson and kendall methods is between 0.3 and 0.7. So, the country's economy indeed affects the performance but not strongly. This analysis made us believe that the longer the athlete stays without achieving a medal, the less possible this athlete will achieve his first medal in the next game.

We achieved this by:
1. Finding the correlation between the number of medals achieved and the GDP
2. Setting up the dataframe to construct the dataframe for correlation
3. Finding the actual kendall and pearson correlations
4. Plotted a scatter chart for the relation between GDP of country and total number of medals achieved



### 2. What is the estimated number of appearances for a participant to achieve their first medal?
We achieved the average number of appearances of 1.575, we infer that the majority took one or two trials only to achieve their first medal, also after using log scale we noticed that the trend is decreasing. Which means less players need more appearances to achieve their first medal.

We achieved this by:
1. Creating an indicator for sorting the dataframe
2. We sort by the year first then by winter and summer giving precedence to winter as it occurs before the summer olympics.
3. We observed the number of athletes achieved at least 1 medal
4. Construct a dictionary containing (key,value)-> (athlete_name,appearances till achieving first medal) --> the dictionary only contains medal holders
5. Average number of appearances is 1.575, this means that medal holders are most likely to achieve their first at an early stage in their career 
6. Use this for the histogram to show stats of the appearances
7. We used those number of appearances and plotted a histogram to represent the counts
8. We used log scale to show how values vary in bins with smaller frequency, we can notice that the trend is decreasing --> which means that less people need more appearances to achieve their first medal.


### 3. What are the features that most affect player performance?
We did not find a single attribute that has a strong direct correlation with performance, we measure performance according to the number of medals attained.
So there was no column independently related to the winning of medals.
Also 3 columns are added to the existing features to support either the logistic or SVM models

Those features are 

1. number of cumulative medals the player got in the last years/Olympic Games  in (commutative_medals)column
2. how many times this player contributes in Olympic Games  described in (commutative_app)
3. if this player team had been in the TOP 5 teams in last years (inTop5)
 
Note: The inTop5 column shows the highest correlation value with medal columns 
we found that inTop5 column could be useful more than the Team column in the prediction process


### 4. Predict whether the player could get a medal or not next game? 
Columns used to predict the medal columns are 
1. Height 
2. Weight
3. NOC 
4. City (could affect winning of the player if this city is in his region )
5. Sport 
6. isSummer 
7. cumulative_app  
8. cumulative_medals2 
9.  inTop5 
10.  BMI  

the columns are chosen based on the correlation values
I used  ColumnTransformer to deal with the categorical Columns like NOC , City and Sport 
I used two different prediction methods first one was Logistic Regression and the seconed was SVM Classifier to predict the Medal Column
Note: the medal column is a binary column that has a value of 1 if this player got any type of medals and 0 otherwise
those two different models are trained on the sorted olympic data beefore 2016 and tested on the 2016 olympic records 
I also used metrics.accuracy_score() to evaluate the two models 
The logistic regression model is used to predict wining of the player in any Sport, the Logistic and SVM models are used to predict wining of the player in specific game  like Athletics and Gymnastics


### 5. How different BMI ranges are demanded, and what are the effects and changes of BMI? (feature engineering 1)
Normal BMI ranges from 18.5 to 24.9. In our dataset, we found out that the BMI of the majority of the athletes fall within the normal age. The percentage of underweight groups was a very minor. There was no significant change in the BMI ranges and mean over the years. Also, the mean BMI was not different between different medal holders with different medals.

We tried observing the BMI along different sports. Again the majority of the sports, their average BMI was within the normal ranges. We had 9 sports which their average BMI was overweight and 1 sport with an underweight BMI.

After extensive research we found out that sports such as the baseball have deceptively high BMIs because of their larger muscles, as such sports favor strength and power training.

On the other hand, the gymnastics BMI is below the normal, again that is because of the nature of the sport, that requires light and flexible body. This is achieved by the gymnast’s diet that keep them small and light weighted. There is also an interesting theory we found that young people undertaking such high levels of gymnastics training are likely to develop overuse injuries and are at risk of causing stress to growing bones and growth plates. A study published in 2004 showed that intense gymnastics training can impact the musculoskeletal growth and maturation that is supposed to occur during puberty.

Last but not least, the weightlifting BMI had the largest range. Again this is due to the nature of the game as athletes compete in a division determined by their body mass, which are 10 categories of different weights for each gender.

We acheived this by:
1. First we observed the BMI histogram to see the BMI distribution
2. We Also did plot the boxplot for the BMI all together
3. finding the number of each category of weight status (underweight - normal - overweight)
4. Plotting corresponding barchart
4. Getting the mean BMI per sport
5. Plotting corresponding barchart for BMI per sport 
6. BMI ranges (boxplot) per each sport
7. Weight status of the means of the BMI per sport
8. Dig deeper into Overweight BMI sports
9. Box plots for BMI of overweighted sports
10. Now for Underweight BMI sports
11. Box plots for BMI of Underweight sports
12. Observing the BMI over the years
13. Observing the BMI for different medal holders
14. BMI over the years for different medals
15. BMI along regions
16. Overweight BMI per region
17. Underweight BMI per region --> none


### 6. How hosting events affects country's performance? (feature engineering 2)
We concluded that indeed on average the country's performance increases significantly when the country hosts an event. We can see from the final figures that the majority of the countries and the events, the number of medals gained increases significantly for the countries who were hosting these events.

We achieved this by:
1. finding total number of unique countries
2. Taking the olympics dataframe and adding a new column Hosting Country of the city hosting the olympics (feature engineering)
3. Creating a dataframe that corresponds to the participants of athletes when their country is the hosting country
4. Creating a dataframe that corresponds to the participants of athletes when their country is **NOT** the hosting country
5. The idea is to compare the average performance of each country when it is hosting vs. when it is not hosting. The metric used is the average number of medals won by each country in both cases(Hosting vs. Not Hosting).
6. Plotted a bar chart for the ratio of average number of medals won by countries (Hosting vs Not-Hosting) to the number of games played --> number of medals/number of times hosted TO number of medals/number of times NOT hosted
7. Plotted a scatter plot for the number of medals won by each country (Hosting vs. Not Hosting)


### 7. Do the number of medals on average increase as an athlete participates more?

This question aims at exploring if countries with high gdp are mainly chosen for hosting the olympics.

1. Remove the countries which has NaN in the GDP column. The only country without GDP is *Western Sahara*
2. Rename some of the countries to match the naming in the athletes dataset.
 United Kingdom -> UK
 United States -> USA
 Korea, South -> South Korea
3. High GDP countries are those which have GDP higher than the third quantile (75th percentile)
4. The results show that out of 22 countries that hosted the olympics, 18 of them have GDP higher than the third quantile.

### 8. Do the number of medals on average increase as an athlete participates more?
We recognised that there is weak/no relationship between the number of participations an athletes does and the number of medals they win. That is unlike what was expected, which was participating more increases the chances of winning a medal. But after plotting the bar graphs and applying the correlation methods we found out the correlation was very weak.

We achieved this by:
1. Getting top 10 players according to the number of medals they gained
2. Plotting corresponding bar graph
2. Gettting the number of games each athelete participated in
3. Getting the total number of medal each athelete won
4. Plotting a bar graph for top 10 participations with their gained medals
5. Plotting another bar graph for top 10 athletes with medals and their corresponding number of participations.
6. Finding correlations using pearson and kendall correlation functions.


### 9. What are the performances of both genders?
Most of these sports have historically been male dominated, number of males participants are more than the double of the number of female participants. Also, it seemed that total number of medals won by males were more than medals won by females. This may sound logical considering the ratio of participants. However the percentage of winning a medal in respect to the gender is almost the same. As well as, male gender tend to participate more in up following events than females, with a ratio of 52 and 45 for males and females, respectively.

The very first Olympic game did not include any female, but there has been a great improvement in female representation ever since. Late 1980s showed the highest increase in the number of female participations.

We achieved this by:
1. Plotting the number of Participants Per Gender
2. Plotting the total number of participations versus total number of medals per gender
3. Plotting the number of Medals Per Gender
4. Finding the percentage of each gender winning medals respect to the number of participants.
5. Plotting the total number of participations versus total number of medals per gender
6. Made a comparison between females vs males for the top two countries
7. Plotting a stacked chart of total medals per Gender over the years.
8. Plotting another stacked chart of each type of medal per Gender over the years.


### 10. Is the number of medals related to the number of each medal?
We found out that as the total number of medals increase, the number of gold, silver, and bronze medals increase separately, which does make sense, that is the positive correlation. The silver medals always had the strongest correlation. But we also recognised that for the performers, the composition of total medals are the highest for gold, we found that the gold medals were gained more, either per region or per athletes, and this was unexpected.

We achieved this by:
1. Getting the regions winning each medal separately
2. Getting the number of all medals (gold+silver+bronze) each region won 
3. Merging the datasets on the region name
4. finding the correlation between total number of medals and each type
5. Exploring the correlation of all performers, not only the top ones.
6. Plotting a scatter chart for each type of medal with total number of medals
7. Plotting the composition of Medals of the top 10 regions
8. Same process as previous, the regions, but for athletes instead. The medals for the top athletes.



# Important note
We submitted the notebook with the cells run already, but if you need to rerun it, please leave the cells for question 3 and 4 at the end, as the model takes time.


# Milestone 3
In this milestone we aimed to create a pipeline for our project/dataset, which we first loaded our datasets from the CSV files. Then, we applied the different preprocessing functions that were applied in the previous milestones, such as cleaning, handling outliers, merging datasets, and feature engineering. Finally we saved our results as CSV files, which is added to our repo named "final_dataset.csv". Note that each of our nodes calls a function that calls other functions which are implemented in milestone 1 or milestone 2. Our node functions are at the start of the file.

We tried to have a clean code and descriptive function and variable names as much as possible. We also tried adding some comments and explaining our functions that may not seem clear, otherwise we felt the name implies what the function/variable/task is for. We have around 30 functions and 11 tasks. 

We created a DAG called olympics_dag that is scheduled to run only once and we used default arguments as shown below. 

dag = DAG(

    'olympics_dag',

    default_args=default_args,

    description='A DAG to extract, transform, load olympics data over 120 years.',

    schedule_interval='@once'

)

default_args = {

    'owner': 'airflow',

    'depends_on_past': False,

    'start_date': datetime(2021, 12, 20),

    'catchup': False

}


**depends_on_past: False** because we want to run our dag even if the last run failed. we don't want dag runs to depend on each other.
**catchUp: False** because we want to run our dag only manually and we don't want to run even if the date is in the past.



## Functions:
1. extract_data: extracting the data for the first two data sets athlete_events and noc_regions
2. prepare_athletes_dataset: removing duplicate rows, handling missing words and fixing the spelling for some words
3. prepare_regions_dataset: renaming countries and regions to a suitable name
4. handle_outliers: here we handle age, weight and height outliers each in a separate function
5. impute_missing_data: here we handle age, weight and height missing values each in a separate function
6. enumerate_medals: in this function we do a label encoding to the medals
7. merge_athletes_regions: merging the two data sets df_olympics and df_regions after data cleaning
8. merge_athletes_countries: extracting the countries of the world.csv data set and cleaning it and merging it to our data set
9. add_BMI_feature: in this function we create the new feature (BMI) and add it to our data.
10. add_Hosting_Country_feature: in this function we create the second new feature (hosting country) and add it to our data.
11. save_data: save the data into a final csv
12. get_age_mean_without_outliers: retireive the mean age without outliers
13. handle_age_outliers: remove outliers from age 
14. handle_weight_outliers: remove outliers from wieght
15. handle_height_outliers: remove outliers from height
16. RemovingWeightOutliersPerGender: handle wieght outliers for each gender seperatly 
17. RemovingHeightOutliersPerGender: handle hieght outliers for each gender seperatly 
18. handle_missing_age: impute the missing age
19. handle_missing_weight: impute the missing weight
20. handle_missing_height: impute the missing height
21. extract_countries: extract countries data set
22. delete_columns: delete unneeded columns
23. get_countries: get an array of all unique countries
24. remove_extra_space: remove extra space in the name of some countries
25. renaming: rename some countries to the correct name
26. change_country_region: change the column country with the name region
27. merge_at_last: merge olypics data set with countries data set
28. all_functions_for_merging: merging the countries of the world.csv and the df_olympics after cleaning both
29. add_hosting_country: hosting country coresponding to each city


## Tasks:
1. extract_data_task: calls extract_data() 
2. preparing_athletes_task: calls prepare_athletes_dataset() 
3. preparing_regions_task: calls prepare_regions_dataset() 
4. outliers_task: calls handle_outliers() 
5. missing_data_task: calls impute_missing_data() 
6. merge_regions_task: calls merge_athletes_regions() 
7. merge_countries_task: calls merge_athletes_countries() 
8. enumerate_medals_task: calls enumerate_medals() 
9. BMI_task: calls add_BMI_feature() 
10. Hosting_Country_task: calls add_Hosting_Country_feature()
11. Save_data_task: calls save_data() 


## Dependencies:
extract_data_task >> preparing_athletes_task >> preparing_regions_task >> outliers_task >> missing_data_task >> enumerate_medals_task >> merge_regions_task >> merge_countries_task >> BMI_task >> Hosting_Country_task >> Save_data_task
![alt text](https://github.com/CSEN1095-W21/project-missing-data/blob/main/ms3dag.jpeg?raw=true)

## Successful Run
![alt text](https://github.com/CSEN1095-W21/project-missing-data/blob/main/olympics%20run.jpeg?raw=true)

