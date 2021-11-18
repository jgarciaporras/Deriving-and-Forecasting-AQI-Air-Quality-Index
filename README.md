# Deriving and Forecasting AQI Air Quality Index

<h3 style="text-align:center"><b>Problem Statement</b></h3>
Exposure to high levels of air pollution can cause a variety of adverse health outcomes. Studies show that people from India are losing 5.2 years of life expectancy. That is why a unit must be introduced which can help identify key indicators in air pollution. AQI is one such unit which measures the concentration of pollutants in the air. It gives an estimate of the Air Quality in a country.


<h3 style="text-align:center"><b>Aim</b></h3>
1. The aim of our project is to derive the Air Quality Index (AQI) and show the correlation between AQI and the different independent features with the help of different Machine Learning Algorithms.
2. Predict the AQI class category based on the features captured by sensors. 
3. Understanding the AQI along with identifying its key indicator can enable governments to take steps to mitigate this problem.
4. 

Data

We got our data from the UCI machine learning repository. The dataset has 9471 instances and 17 features.  It contains the concentration or volume of various pollutants such as CO, C6H6, NO, NO2 found in the air in Italy. These readings were recorded using different sensors. Apart from this the features also included the temperature, humidity along with the date and time at which they were recorded.
 

Source: https://archive.ics.uci.edu/ml/datasets/Air+quality

Data Preprocessing

 We start By Importing different libraries and reading the dataset. 
 
After reading the dataset and displaying top 20 observations, we found the following issues -
 

 Solutions - 
•	To begin, we first dropped columns that were not contributing to the prediction (eg: Date, Time). 
•	Replaced "," by "." 
•	Replaced all the missing values (-200 and –200.0) by "nan" instead of dropping
•	Used a SimpleImputer with mean to impute the nan values.
•	Finally,  changed the data type to float

 

In the below image, we can see that –200 value is being replaced with nan  values.
 

Now we can Impute the Nan values with 
 

Here is the final output after dropping down the columns, replacing errors with Nan value and then imputing those values. Here only 5 attributes remains after performing all those operations.
 

                                                     
                                                        Feature Selection and Engineering

Here, we are creating a new attribute named as AQI which we will calculate through these 5 attributes like CO, NMHC, C6H6, Nox and No2 
 
 
 

Here, we are displaying top 5 samples of our dataset and we can see that a new attribute named AQI is being formed.
 

Based on the AQI scores, we are making a new attribute namely AQI Category.
 
Then we display top 5 observations of our dataset
 

Data Imbalance

 
After binning and deriving the class labels from the features we found that the there was severe imbalance in the data. The bulk of the data i.e close to 78% belonged to either “Unhealthy for sensitive groups” and “Moderate”. The last three classes had only around 500 instances which would make it strenuous for the model to effectively learn these classes.

Upon fitting a RandomForestClassifier to get a rough baseline we found that the model had an accuracy of 91%. But on checking the classification report we noticed that “Hazardous” had a f1-score of 0 and likewise classes like “Unhealthy” and “Very unhealthy” too had comparatively low scores.
  

To tackle this, we explored an oversampling technique called SMOTE (Synthetic Minority Oversampling Technique). This takes the minority classes and creates artificial rows which are similar to the original values and brings them on level with the majority class.
 
So after fitting another RandomForest classifier with a stratified train test split we found that the f1-scores or the labels had improved drastically.

However, there was still room for improvement.

Pipeline

After running our model with RandomForestClassifier, we obtained the results above (Image 1): 
 
Image 1: Classification Report
So, we will know if that model it is the best for our dataset.
We will create pipeline because it helps us to process data in sequence which it means that the result from one segment become the input from the next one and so on. 
The purpose of the pipeline is to assemble several steps that can be cross validated together while setting different parameters. 
Steps that we need to do:
1.	Importing the libraries needed:

 
Image 2: Libraries needed

2.	Creating our models and pipeline: To determine that our Random Forest Classifier is the best model, we create Logistic Regression, Support Vector Machine, Random Forest Classifier with different parameters, K-Nearest-Neighbor and Ada Booster (Image 3)
 
Image 3: Libraries needed


3.	Furthermore, we create a Repeat Stratified K Fold which help us to improve the performance of the estimation. This part of the code will be used on the GridSearchCV.
 
Image 4: Creating Repeat Stratified K Fold


Model Selection and Hyperparameter tuning using GridSearchCV
 
Questions?
•	Is our Random Forest Classifier the best model for our datasets?
•	How about others with Hyperparameter tunning?

We wanted to figure out which one gives us the best performance for our dataset.  In order to get answers for these question, we decided to use GridSearchCV as below.
GridSearchCV for finding the best optimal combination of hyperparameters for other models: 
 

Our GridSearchCV applies a ‘fit’ and a ‘score’ method as well as a ‘predict’ using the scaled train data and defined models in the pipelines. I used a simple “for” loop with GridSearchCV which tries each of the classifiers one by one with the corresponding parameter grid specified in the dictionary. Also, to check if the model is overfitting, we added predict with test data too.  

 




The results show 

•	Our Random Forest Classifier model takes 2nd place with the hyperparameter ( n_estimators=200,  max_depth = 8 ). It gave us 0.982894 Test score.
•	The best model is SVC (C = 20) with 0.996551 score Test score. 

 


  




Model Evaluation

After running our Grid Search and creating hyperparameters, we can compare the Matrix Confusion from our first Random Forest Classifier(Image 5)   and the best Random Forest Classifier(Image 6) . This allows us to see how improve the accuracy.
 
Image 5: Confusion Matrix – 1st Random Forest Classifier

			

 
Image 6: Confusion Matrix – Best Random Forest Classifier


On evaluating the final model we found that the f1-score of classes like “Hazardous”, “Unhealthy” and “Very Unhealthy” have improved drastically. Along with this there was a big improvement in the overall accuracy and f1-score after hyperparameter tuning.
 


 
Correlation Heatmap

 

To understand the key indicators in predicting the AQI we plotted a Pearson correlation heatmap. Features/ pollutants such as CO, NO, and NO2 have a strong correlation with AQI.


Conclusion

To conclude, we derived the Air Quality Index (AQI) from the concentration of pollutants by averaging the hourly volume of pollutants in the air and binning them into respective categories. Solved data imbalance by oversampling using SMOTE. The final RandomForest model after oversampling and hyperparameter tuning gave an accuracy of 0.9833 with consistent f1-scores for each label. 
Key indicators which contribute towards AQI are CO, NO, and NO2. Governments can use these key indicators to help understand the source of the problem and take steps towards mitigating air pollution.

For future work can include date and time to forecast the AQI.





