from datetime import datetime
from datetime import date
import requests
import json
import tweepy # for tweets api
from tweepy import OAuthHandler
from textblob import TextBlob #sentiment analysis


import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.docker_operator import DockerOperator

import pandas as pd


default_args= {
    'owner':'airflow',
    'depends_on_past': False,
    'start_date': datetime(2021, 12, 27),
    'catchup': True
}

dag = DAG(
    'twitter_bonus',
    default_args=default_args,
    description='Fetch covid data from api',
    schedule_interval='@daily',
)

def retrieve_tweets(ds, ti):
    search_words="*" #any tweets
    #exclude Links,retweets,replies
    search_query = search_words + " -filter:links AND -filter:retweets AND -filter:replies" 
    #current date
    tweets1=get_tweets(search_query,"eg",ds)
    tweets2=get_tweets(search_query,"us",ds)

    tweets_dict= {
        "tweets1":tweets1,
        "tweets2":tweets2,
    }
    ti.xcom_push(key='tweets_retrieval', value=tweets_dict)


def get_tweets(search_query,country,date_until):
    #tokens used
    consumer_key = "PpotiraZv0MGKHj1T8POHGo45"
    consumer_secret = "MZXYCPLSckwqQ3RAZ4XVRoetwioBrr4bkg4AOaS9fYzRF24XLg"
    access_token = "1178023920504643584-J7xJ7o6NPTvk9n5vqCwIeQ9j9dbwak"
    access_token_secret = "xFu5OBeaEguRMw1W4SufJbDm6X30fgyz2yqNbRlKnkvyt"

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)
    #verifying the user
    print(api.verify_credentials().screen_name)

    # Collect tweets using the Cursor object (performs the api request with the tunable parameters)
    geo="26.820553,30.802498,1000km"
    if country == "us":
        geo="37.09024,-95.712891,1000km"
    #lat and long were achieved online for the desired countries, 1000km is the radius of search
    tweet_list = [tweets for tweets in tweepy.Cursor(api.search_tweets,
                                    q=search_query,
                                    lang="en",
                                    geocode=geo,
                                    until=date_until,
                                    tweet_mode='extended').items(20)]
    #print(tweet_list2)
    tweets_text=[]
    for tweet in tweet_list[::-1]:#extracting the list containing details about the received tweets
        tweets_text.append(tweet.full_text)
    return tweets_text


def getPolarity(text):
    return TextBlob(text).sentiment.polarity


#returns text for the country (per 20 tweets) *to be saved in the csv
def compare(average_polarity,performance,country_name):
    #performance is 0 for bad, 1 for good
    performed="good"
    if performance==0:
        performed="bad"
    match="analysis not matching"
    if average_polarity<=0 and performance==0:
        match="analysis matching"
    if average_polarity>0 and performance==1:
        match="analysis matching"
    
    text = "Country "+country_name+" performed "+ performed +", and the average polarity is "+str(average_polarity)+", Conclusion: "+match+"."
    return text

def sentiment_analysis(ti):

    tweets_dict = ti.xcom_pull(key='tweets_retrieval', task_ids='retrieve_tweets')
    #getting array of polarities of each tweet in the tweets array of each country
    polarities1=[]
    polarities2=[]
    tweets1 = tweets_dict['tweets1']
    tweets2 = tweets_dict['tweets2']
    print(tweets1)
    for i in range(len(tweets1)):
        polarities1.append(getPolarity(tweets1[i]))
    for i in range(len(tweets2)):
        polarities2.append(getPolarity(tweets2[i]))

    polarities_dict = {
        "polarities1":polarities1,
        "polarities2":polarities2
    }
    ti.xcom_push(key='sentiment', value=polarities_dict)


def average_polarity(ti):
    
    polarities_dict = ti.xcom_pull(key='sentiment', task_ids='sentiment_analysis')

    #calculating average polarity for each country
    average_polarity1=0
    average_polarity2=0
    print(polarities_dict['polarities1'])
    for i in range(len(polarities_dict['polarities1'])):   
        average_polarity1+=polarities_dict['polarities1'][i]
    average_polarity1/=len(polarities_dict['polarities1'])
    
    for i in range(len(polarities_dict['polarities2'])):
        average_polarity2+=polarities_dict['polarities2'][i]
    average_polarity2/=len(polarities_dict['polarities2'])

    polarities_dict = {
        "average_polarity1":average_polarity1,
        "average_polarity2":average_polarity2
    }

    ti.xcom_push(key='avg_polarities', value=polarities_dict)

def compare_both_countries(ds, ti):
    # writes into a file the comparison of actual performance and the polarity achieved from sentiment analysis 
    # TODO ADD CURRENT DATE
    avg_dict = ti.xcom_pull(key='avg_polarities', task_ids='average_polarity')
    date = ds
    with open('./transformations/Analysis_'+date, 'w') as f:
        f.write(compare(avg_dict['average_polarity1'],0,"Egypt"))
        f.write('\n')
        f.write(compare(avg_dict['average_polarity2'],1,"United States"))
        f.write('\n')
        f.write("TIMESTAMP: "+date)
        f.write('\n')


tweets_retrieval_task = PythonOperator(
    task_id='retrieve_tweets',
    python_callable=retrieve_tweets,
    provide_context=True,
    dag=dag
)

sentiment_analysis_task = PythonOperator(
    task_id='sentiment_analysis',
    python_callable=sentiment_analysis,
    provide_context=True,
    dag=dag
)

polarity_task = PythonOperator(
    task_id='average_polarity',
    python_callable=average_polarity,
    provide_context=True,
    dag=dag
)

comparison_task = PythonOperator(
    task_id='compare_both_countries',
    python_callable=compare_both_countries,
    provide_context=True,
    dag=dag
)



tweets_retrieval_task >> sentiment_analysis_task >> polarity_task >> comparison_task