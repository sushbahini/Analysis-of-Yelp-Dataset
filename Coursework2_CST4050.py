#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction:

# Yelp is a public company which publish crowd-sourced reviews about local businesses, as well as the online reservation service Yelp Reservations. It also assists small business in how to respond to reviews. The company was founded in the year 2004 by former PayPal employees. By the year 2010 it had $30 million in revenues and the website had published more than 4.5 million crowd-sourced reviews. 

# # 2. Dataset Despriction

# The data used here is collected from Yelp website (https://www.yelp.com/dataset). The data was uploaded there for learning/academic purpose. The data consists of many JSON files. There are 6 JSON files available in the website. For my project I have used 2 of them. One file consists of the business data. It contains many vital business data. The other file used in the analysis consists of the reviews received from the users. The details of the attributes of these two files are listed below:
# 
# Business Data: 
# 
# "business_id": It is a unique string string business id.
# 
# "name": It is a string charater. It is name of the business
# 
# "address": It is a string character, It is the full address of the business.
# 
# "city": It is a string character, It is consists of the city where the business is located.
# 
# "state": It is a string character, It consists of the character state code, if applicable.
# 
# "postal code": It is a string character, It consists of the postal code of the location of the business
# 
# "latitude": It is a float character. It is the latitude of the location.
# 
# "longitude": It is a float character. It is the longitude of the location.
# 
# "stars": It is a float character. It is consists of the star rating, rounded to half-stars.
# 
# "review_count": It is a integer character, it consists of number of reviews.
# 
# "is_open": It is an integer, 0 or 1 for closed or open, respectively.
# 
# "attributes": It is an object, business attributes to values. note: some attribute values might be objects
# 
# "categories": It is an array of strings of business categories. 
# 
# "hours": It is an object of key day to value hours, hours are using a 24hr clock.
# 
# Review Data:
# 
# "review_id": It us a string character. It is an unique review id.
# 
# "user_id": It is a string character. It is an unique user id that maps to the user in user.json.
# 
# "business_id": It is a string character. It is a business id that maps to business in business.json.
# 
# "stars": It is an integer that consists of the star rating.
# 
# "date": It is a string character which is in date formatted YYYY-MM-DD.
# 
# "text": It is a string character which consists of the reviews itself.
# 
# "useful": It is a integer which consists of the number of useful votes received.
# 
# "funny": It is a integer which consists of number of funny votes received.
# 
# "cool": It is a integer which consists of number of cool votes received. 
# 

# In[231]:


# Loading the required libraries

# numpy is the library imported for doing the linear algebra
import numpy as np 

# pandas are used for data processing, JSON file I/O (e.g. pd.read_JSON)
import pandas as pd 

import collections # this is used to store collections of data
import re, string # this is used for string searching and manipulation
import sys # the sys module provides information about constants, functions and methods of python interpreter. 
import time 


# In[232]:


# The below function loads the JSON file and converts it into a dataframe in pandas. 
import json

def init_ds(json):
    ds= {}
    keys = json.keys()
    for k in keys:
        ds[k]= []
    return ds, keys

def read_json(file):
    dataset = {}
    keys = []
    with open(file,encoding="utf8") as file_lines:
        for count, line in enumerate(file_lines):
            data = json.loads(line.strip())
            if count ==0:
                dataset, keys = init_ds(data)
            for k in keys:
                dataset[k].append(data[k])
                
        return pd.DataFrame(dataset)


# In[3]:


yelp_business= read_json('yelp_academic_dataset_business.json')# The business data file
yelp_review= read_json('yelp_academic_dataset_review.json') # The review data file


# First glance at the review file

# In[5]:


yelp_review.head()


# First glance at the business file

# In[6]:


yelp_business.head()


# # 3. Machine learning challenge 

# The challenge here is to build a model to clasify the Yelp Reviews into 1 star, 3 star or 5 star(Negative,average,good) categories based off the text content.
# We can also use the model for Sentiment Analysis and Prediction of Review Ratings. 
# 
# This can be achieved by performing machine learning for "textual data analysis" . This allows to extarct and classify the reviews to make better predictions and create insights.  

# # 4. Better Understanding of the data

# Before preparing the required dataset, I want to analyse what will be the best parameter to filter the data.    

# # -- Top reviwed busines(by name)

# In[233]:


top_reviewed = yelp_review[yelp_review["stars"]>3]
top_reviews_dict ={}

for business_id in top_reviewed["business_id"].values:
    try :
        top_reviews_dict[business_id] =top_reviews_dict[business_id]+1
    except:
        top_reviews_dict[business_id]=1
        
topbusiness = pd.DataFrame.from_dict(data= top_reviews_dict,orient="index")

topbusiness.reset_index(inplace=True)
topbusiness.columns = ['business_id', 'rated']
del(top_reviews_dict)
del(top_reviewed)


# In[234]:


top_count= 20
right=pd.DataFrame(yelp_business[['business_id',"name","categories"]].values,
                    columns=['business_id',"Business name","categories"])

top_business_data = pd.merge(topbusiness,right=right, how="inner",on='business_id')
top_business_data.sort_values("rated")[::-1][:top_count].plot(x="Business name",y="rated", 
                                                   kind="bar",figsize=(14,6),
                                                   title='Positive reviews').set_ylabel("Total ratings")

del(topbusiness)
del(right)


# The top rewiwed business merchants are Mon Ami Gabi, Bacchanal Buffet, Hash House A Go Go. All these are restuarants. 

# # -- Categories of top reviewed businesses

# In[9]:


num_cat =10 # to show top 10 catrgories
top_business = 30 # choose categories of top 30 businesses
cat_data = top_business_data.sort_values("rated")[::-1][:top_business]
#cat_data.categories
Categories={}
for cat in cat_data.categories.values:
    all_categories= cat.split(",")
    for x in all_categories:
        try :
            Categories[x] =Categories[x]+1
        except:
            Categories[x]=1
top_categories = pd.DataFrame.from_dict(data= Categories,orient="index")
top_categories.reset_index(inplace=True)
top_categories.columns = ['category', 'occurance']

x_val=top_categories.sort_values("occurance")[::-1][:num_cat].occurance.values
labels=top_categories.sort_values("occurance")[::-1][:num_cat].category.values
series = pd.Series(x_val, index=labels, name='Top business types')
series.plot.pie(figsize=(10, 10),startangle=90)


# Restuarants are the top reviwed business categories. 

# # -- Negatively reviewed businesses

# In[10]:


bottom_reviewed = yelp_review[yelp_review["stars"]<2]
bottom_reviews_dict ={} 

for business_id in bottom_reviewed["business_id"].values:
    try :
        bottom_reviews_dict[business_id] =bottom_reviews_dict[business_id]+1
    except:
        bottom_reviews_dict[business_id]=1
        
bottombusiness = pd.DataFrame.from_dict(data= bottom_reviews_dict,orient="index")

bottombusiness.reset_index(inplace=True)
#bottombusiness.head()
bottombusiness.columns = ['business_id', 'rated']


# In[11]:


top_count= 20
right=pd.DataFrame(yelp_business[['business_id',"name","categories"]].values,
                    columns=['business_id',"Business name","categories"])

bottom_business_data = pd.merge(bottombusiness,right=right, how="inner",on='business_id')
bottom_business_data.sort_values("rated")[::-1][:top_count].plot(x="Business name",y="rated", 
                                                   kind="bar",figsize=(14,6),
                                                   title='Negative reviews').set_ylabel("Total 1 star ratings")

del(bottom_reviewed)
del(bottom_reviews_dict)
del(bottombusiness)
del(right)


# Casinos are the negatively reviwed business.

# # 5. Methodology

# # 5.1 Data Collection

# From the above analysis, it is clear that the restaurants are the top reviwed business.Therefore, for sentimental analysis I want to collect the reviews for "Indian" restaurants only. 
# 
# I will merge the two datafiles now. I want to develop the rating predictive model for the "Indian" restaurants only based on their reviews. 

# Since I want to collect the reviews only for the "Indian restaurants", I have to extract the category column as individual elements. So that I can filter my data accordingly. Further I want to collect the data for the Indian restaurants which are still open.  

# In[12]:


# The explode() function is used to transform each element of a list-like to a row. 
# To use this function, pandas version needs to be above 0.25. 
# Checking the version of pandas. 
pd.__version__


# The explode() function is used to transform each element of a list-like to a row

# In[13]:


# Applying the explode function to the column "categories"
df_explode = yelp_business.assign(categories = yelp_business.categories
                         .str.split(', ')).explode('categories')


# In[14]:


#We can then list out all the individual category
df_explode.categories.value_counts()


# I want to get the reviews for the Indian Restaurants

# In[15]:


#Find the categories containing Indian Restaurants
df_explode[df_explode.categories.str.contains('Indian',
                      case=True,na=False)].categories.value_counts()


# In[16]:


print('Total number of categories',len(df_explode.categories.value_counts()))


# In[17]:


# Finding the categories that contains Restaurants
df_explode[df_explode['categories'].str.contains('Indian',case=True,na=False)].categories.value_counts()


# In[18]:


# Extracting the data only for the Indian Restaurants from the buiness file.
business_Restaurants = yelp_business[yelp_business['categories'].str.contains(
              'Indian',case=False, na=False)]


# In[19]:


business_Restaurants.head()


# In[20]:


# Collecting the data only for the Indian restaurants which are still open. 
# 1 = open, 0 = closed
business_Restaurants = business_Restaurants[business_Restaurants['is_open']==1]


# In[21]:


# Remove the columns that are not required in business_Restaurants
drop_columns = ['city','state','postal_code','latitude','longitude','hours']
business_Restaurants= business_Restaurants.drop(drop_columns, axis=1)
business_Restaurants.head()


# Review data file is a huge file. If we will try to load all the data at once, it is likely to crash the memory of the computer. Therefore we will load large data by segmenting the file into smaller chunks. Also to reduce the memory usage I am identifying the datatype of each column. 

# In[22]:


# To reduce the memory usage identifying the datatype of each column
size = 100000
review = pd.read_json('yelp_academic_dataset_review.json', lines=True,
                      dtype={'review_id':str,'user_id':str,
                             'business_id':str,'stars':int,
                             'date':str,'text':str,'useful':int,
                             'funny':int,'cool':int},
                      chunksize=size)


# In[23]:


# There are multiple chunks to be read
chunk_list = []
for chunk_review in review:
    # Drop columns that aren't needed
    chunk_review = chunk_review.drop(['review_id','date'], axis=1)
    # Renaming column name to avoid conflict with business overall star rating
    chunk_review = chunk_review.rename(columns={'stars': 'review_stars'})
    # Inner merge with edited business file so only reviews related to the business remain
    chunk_merged = pd.merge(business_Restaurants, chunk_review, on='business_id', how='inner')
    # Show feedback on progress
    print(f"{chunk_merged.shape[0]} out of {size:,} related reviews")
    chunk_list.append(chunk_merged)
# After trimming down the review file, concatenate all relevant data back to one dataframe
df = pd.concat(chunk_list, ignore_index=True, join='outer', axis=0)


# **We have finally collected our required data for the application of our machine learning models (Sentimental analysis)**.

# In[24]:


# To view the collected data
df.head()


# # 5.2 Data Analysis Process

# # 5.2.1 Data Preprocessing

# The preprocessing of the data includes removing the unwanted columns from the dataset. It also includes eliminating stopwords and punctuations. This is achieved by using nltk in python. It stands for natural language Toolkit. It is a suite of libraries and programs for symbolic and statistical natural language processing. I have also created a new column in the named as "text length". It returns the number of words in the column "text". Since, text data requires special preparation before we start it for predictive modeling, text must be parsed to remove words, called tokenization. For the input to a machine learning algorithm, the words need to be encoded as integers or floating point values. This is called feature extraction (or vectorization).

# In[25]:


# To get the information about the object types in the new data.       
df.info()


# In[26]:


# To get a statics information on the numerical column of the data
df.describe()


# In[27]:


# Removing the un-wanted columns from the dataset
drop_columns = ['name','address','stars','review_count','is_open','attributes','categories']
df= df.drop(drop_columns, axis=1)


# In[28]:


# Creating a new column which gives the number of words in the text column
df['text length'] = df['text'].apply(len)
df.head()


# In[29]:


# To check if there is any missing data in the new dataFrame
df.isnull()


# # 5.2.2 Data Visualisation

# In[30]:


# Importing the libraries to visualise the data
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[31]:


# Using FacetGrid from the seaborn library created a grid of 5 histograms of text length based on the star ratings
g = sns.FacetGrid(df,col='review_stars')
g.map(plt.hist,'text length')


# **The above graph shows that the more is the review_stars the lenght of the text lies between 0-1000**  

# In[47]:


# Counting the number of occurrences for each type of star rating
sns.countplot(x='review_stars',data=df,palette='rainbow')


# **The above graph shows that the Indian restaurants have recieved more positive reviews** 

# In[48]:


# calculating the mean values of the numerical columns, grouping it by the category, stars
stars = df.groupby('review_stars').mean()
stars


# In[49]:


# Visualising the correlation between the dataframe stars
sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)


# # 5.2.3 Sentiment Detection

# In[37]:


# Classifying the dataset and splitting it into the reviews and stars.
# Here, we will classify the dataset into 3 types of rating. Rating 1 = "negative" , 3 ="Average", and 5 ="Positive".
data_class = df[(df.review_stars==1) | (df.review_stars==3) | (df.review_stars==5)]


# In[38]:


# Creating the feature and target. x is the 'text' column of data_class and y is the 'stars' column of data_class.
X = data_class['text']
y = data_class['review_stars']
print(X.head())
print(y.head())


# # 5.2.4 Preparing the data for predictive Modelling

# Below I have defined a function that will clean the dataset by removing stopwords and punctuations. nltk stands for natural language Toolkit. It is a suite of libraries and programs for symbolic and statistical natural language processing.

# In[99]:


import nltk
from nltk.corpus import stopwords

# CLEANING THE REVIEWS - REMOVAL OF STOPWORDS AND PUNCTUATION
def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[101]:


# Using the fit_transform method on the CountVectorizer object and passing the 'text' column. Saved the result by overwriting x.
vocab = CountVectorizer(analyzer=text_process).fit(X)
x = vocab.transform(X)


# # 5.3 Train Test Split 

# In order to apply the models, we have to split the data into train and test data. Here, I have divided the data into 70-30. I have taken 70% of the data as train and remaining 30% of the data as test.  

# In[103]:


from sklearn.model_selection import train_test_split


# In[104]:


x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3,random_state=101)


# # 5.4 Applying the Machine Learning Models

# In this case, we are applying classification learning algorithms. It is a supervised learning process. It is a process where the computer program learns from from the input and then uses this to classify new observations. 
# 
# The basic idea for opting a classification model is when the target variable is categorical. In this case, we are trying to predict the rating of a review. We have got 5 ratings(0,1,2,3,4,5) out of which we need to predict one. Therefore, classification models can help us in this. 
# 
# In my analysis, I have used Naive Bayes Classifier, Random Forest Classifier, Decision Trees, K-Nearest Neighbor, RNN Model from Neural Networks. 
# 

# # 5.4.1 Training a Model using Naive Bayes classifier

# Firstly, I want to establish a model that will act as a baseline. In general terms a linear model is appropriate and has the advantage of being fast to train.
# 
# Here, I have used Multinomial Naive Bayes over Gaussian because with a sparse data, Gaussian Naive Bayes assumptions aren't met. A simple gaussian fit over the data will not give us a good fit or prediction. 
# 
# Naive Bayes classification technique is based on Bayes’ Theorem with the assumption of independence among predictors. This classifier is easy to build. It is known to be a good fit for very large datasets due to its high scalability. 

# In[105]:


# Importing the required libraries.
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report


# In[106]:


# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_train,y_train)
predmnb = mnb.predict(x_test)
print("Confusion Matrix for Multinomial Naive Bayes:")
print(confusion_matrix(y_test,predmnb))
print("Score:",round(accuracy_score(y_test,predmnb)*100,2))
print("Classification Report:",classification_report(y_test,predmnb))


# **The performance score of Naive Bayes classifier is 86.06. Since it is high score, I will treat this model as my baseline.**

# # 5.4.2 Random Forest Classifier

# There is no correlation between our feature(text) and target(review_stars) and this is the reason for choosing Random Forest Classifier.
# The vital thing for a Random Forest Classifier model to make an accurate class prediction is the trees of the forest and more importantly their predictions need to be uncorrelated (or at least have low correlations with each other).
# 
# Random forests are an ensemble learning method for classification. It operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

# In[107]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
rmfr = RandomForestClassifier()
rmfr.fit(x_train,y_train)
predrmfr = rmfr.predict(x_test)
print("Confusion Matrix for Random Forest Classifier:")
print(confusion_matrix(y_test,predrmfr))
print("Score:",round(accuracy_score(y_test,predrmfr)*100,2))
print("Classification Report:",classification_report(y_test,predrmfr))


# **The performance score of Random Forest Classifier is 77.9.** 

# # 5.4.3 Decision Tree

# Decision Tree are of two types a) classification b) regression. Since the decision variable(target) is categorical/discrete we will be using decision tree classifier. It builds the model in the form of tree structure. The classifier breaks down a data set into smaller subsets and at the same time an associated decision tree is incrementally developed. Decision trees can handle both categorical and numerical data.  

# In[108]:


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
preddt = dt.predict(x_test)
print("Confusion Matrix for Decision Tree:")
print(confusion_matrix(y_test,preddt))
print("Score:",round(accuracy_score(y_test,preddt)*100,2))
print("Classification Report:",classification_report(y_test,preddt))


# **The performance score of Decision Tree model is 75.62**

# # 5.4.4 K Nearest Neighbour Algorithm

# KNN is known as a non-parametric and lazy learning algorithm. It is a supervised classification technique that uses proximity as a proxy for ‘sameness’. This algorithm takes a bunch of labelled points and uses them to learn how to label other points. To label a new point, it looks at the labelled points closest to that new point (those are its nearest neighbors). Closeness is typically expressed in terms of a dissimilarity function. Once it checks with ‘k’ number of nearest neighbors, it assigns a label based on whichever label the most of the neighbors have.

# In[109]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)
predknn = knn.predict(x_test)
print("Confusion Matrix for K Neighbors Classifier:")
print(confusion_matrix(y_test,predknn))
print("Score: ",round(accuracy_score(y_test,predknn)*100,2))
print("Classification Report:")
print(classification_report(y_test,predknn))


# **The performance score of K Neighbors Classifier is 68.74**

# # 5.4.5 RNN Model

# The RNN is an expressive model that is known to learn highly complex relationships from an arbitrarily long sequence of data. It maintains a vector of activation units for each element in the data sequence, this makes RNN very deep. The depth of RNN leads to two well-known issues, the exploding and the vanishing gradient problems. 
# 
# There are many ways to implement nueral network in python. Here, I will be using tensorflow/keras.

# In[110]:


# Importing the libraries
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[111]:


tk = Tokenizer(lower = True)
tk.fit_on_texts(X)
X_seq = tk.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=100, padding='post') 


# In[112]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size = 0.25, random_state = 1)


# In[113]:


batch_size = 64
X_train1 = X_train[batch_size:]
y_train1 = y_train[batch_size:]
X_valid = X_train[:batch_size]
y_valid = y_train[:batch_size]


# In[114]:


from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
vocabulary_size = len(tk.word_counts.keys())+1
max_words = 100
embedding_size = 32
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(200))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[115]:


model.fit(X_train1,y_train1,validation_data=(X_valid,y_valid),batch_size=batch_size,epochs=10)


# **The performance score of RNN Model is 26.56**

# # 6. Score of the above classifier models-- Results

# Multinomial Naive Bayes -- 86.06,
# Random Forest Classifier -- 77.9,
# Decision Tree -- 75.62,
# K Nearest Neighbour Classifier -- 68.74,
# RNN -- 26.56.

# Multinomial Naive Bayes has the best score. We will use this model to predict a random positive, negative and average review.

# # 7. Validation

# # 7.1 Predict positive review

# In[134]:


# POSITIVE REVIEW
pre = df['text'][20]
print(pre)
print("Actual Rating: ",df['review_stars'][20])
pre_t = vocab.transform([pre])
print("Predicted Rating:")
mnb.predict(pre_t)[0]


# # 7.2 Predict Average Review

# In[235]:


# AVERAGE REVIEW
ar = df['text'][6]
print(ar)
print("Actual Rating: ",df['review_stars'][6])
ar_t = vocab.transform([ar])
print("Predicted Rating:")
mnb.predict(ar_t)[0]


# # 7.3 Predict Negative Review

# In[229]:


# NEGATIVE REVIEW
nr = df['text'][58]
print(nr)
print("Actual Rating: ",df['review_stars'][10])
nr_t = vocab.transform([nr])
print("Predicted Rating:")
mnb.predict(nr_t)[0]


# # 8. Summary:
# 

# From the datasets, we have found that:
# 1 -- Mon Ami Gabi is the merchant who has got the maximum number of positive reviews.
# 2 -- The category of top most reviwed business is restaurants.
# 3 -- Casinos have got the most negative reviews. 
# 
# From the machine learning models for sentimental analysis, it is clear that Multinomial Naive Bayes performes the best.
# The validation proves that the model works fine for positive and average reviews. But it seems to be not working for the negative reviews. 

# # 9. Future Scope:

# I believe the reason for which the model fails to predict the negative review is that there are more positive reviews as compared to the negative ones in the dataset(the collected data,df). That means the dataset is not normally distributed. I can suggests two ways to improve it:
# 
# 1 -- We can normalize the data. So that the positive and negative ratings are equally distributed over the dataset.
# 
# 2 -- While collecting the data, we can also check with other business categories (Shopping, food, home services,etc). It misght be possible that only the reviews for "Indian Restaurants" contains mostly positive ones.  
