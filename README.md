# Deriving and Forecasting AQI (Air Quality Index)

<h3 style="text-align:center"><b>Problem Statement</b></h3>
Exposure to high levels of air pollution can cause a variety of adverse health outcomes. Studies show that people from India are losing 5.2 years of life expectancy. That is why a unit must be introduced which can help identify key indicators in air pollution. AQI is one such unit which measures the concentration of pollutants in the air. It gives an estimate of the Air Quality in a country.


<h3 style="text-align:center"><b>Aim</b></h3>
1. The aim of our project is to derive the Air Quality Index (AQI) and show the correlation between AQI and the different independent features with the help of different Machine Learning Algorithms.
2. Predict the AQI class category based on the features captured by sensors. 
3. Understanding the AQI along with identifying its key indicator can enable governments to take steps to mitigate this problem.
4. 

Data

We got our data from the UCI machine learning repository. The dataset has 9471 instances and 17 features.  It contains the concentration or volume of various pollutants such as CO, C6H6, NO, NO2 found in the air in Italy. These readings were recorded using different sensors. Apart from this the features also included the temperature, humidity along with the date and time at which they were recorded.

![data](https://user-images.githubusercontent.com/44596318/142511717-b231b379-8c3a-45e0-8ab8-24898e612aa6.jpg)


Source: https://archive.ics.uci.edu/ml/datasets/Air+quality

Data Preprocessing

 We start By Importing different libraries and reading the dataset. 
 
After reading the dataset and displaying top 20 observations, we found the following issues -

![Picture2](https://user-images.githubusercontent.com/44596318/142511719-d1f4bb99-cebb-4a6c-b1c4-3a21bd5cdc13.jpg) 

 Solutions - 
•	To begin, we first dropped columns that were not contributing to the prediction (eg: Date, Time). 
•	Replaced "," by "." 
•	Replaced all the missing values (-200 and –200.0) by "nan" instead of dropping
•	Used a SimpleImputer with mean to impute the nan values.
•	Finally,  changed the data type to float

![Picture3](https://user-images.githubusercontent.com/44596318/142511720-55440b6c-0961-49f1-b55d-01fcde4bc9bd.jpg) 

In the below image, we can see that –200 value is being replaced with nan  values.

![Picture4](https://user-images.githubusercontent.com/44596318/142511722-dd3e6ff1-2059-448e-8d7f-51a55e16d52f.jpg) 

Now we can Impute the Nan values with 

![Picture5](https://user-images.githubusercontent.com/44596318/142511723-b34dd1dd-7ad5-4c87-aab7-1b484b922246.jpg)
 

Here is the final output after dropping down the columns, replacing errors with Nan value and then imputing those values. Here only 5 attributes remain after performing all those operations.

![Picture6](https://user-images.githubusercontent.com/44596318/142511724-552d0ac4-84ad-499a-b0a7-8476bc66704f.jpg)
 

                                                     
                                                        Feature Selection and Engineering

Here, we are creating a new attribute named as AQI which we will calculate through these 5 attributes like CO, NMHC, C6H6, Nox and No2  
 
![Picture7](https://user-images.githubusercontent.com/44596318/142511725-15ee58f5-6999-434b-b7d6-1d38a5ae3a5f.jpg)

![Picture8](https://user-images.githubusercontent.com/44596318/142511726-e1c33d56-1332-478a-a4fb-fce6462b83c6.jpg) 

Here, we are displaying top 5 samples of our dataset and we can see that a new attribute named AQI is being formed.

![Picture9](https://user-images.githubusercontent.com/44596318/142511728-1a5c3d12-d0c9-4d60-a718-0fa3d6cace6c.jpg) 

Based on the AQI scores, we are making a new attribute namely AQI Category.


![Picture10](https://user-images.githubusercontent.com/44596318/142511729-61eb36bc-413c-429f-aed5-cea1326d6fe3.jpg)
 
Then we display top 5 observations of our dataset

![Picture23](https://user-images.githubusercontent.com/44596318/142511711-1a8f4e59-3651-483f-96c4-54b5a0a4aea4.jpg)
![best_model](https://user-images.githubusercontent.com/44596318/142511713-f679962f-b0a3-41ec-9432-6498e9e3c12c.png)
![best_model_plot](https://user-images.githubusercontent.com/44596318/142511714-735db16c-ff98-462b-85b6-9217998b2f9f.png)
![confusion_matrix](https://user-images.githubusercontent.com/44596318/142511716-a148dffb-88f1-4e66-9d4a-e30601c62a1b.jpg)
![Picture11](https://user-images.githubusercontent.com/44596318/142511731-dbad003f-485e-4ffa-ab57-6505d8e1d6c9.jpg)
![Picture12](https://user-images.githubusercontent.com/44596318/142511732-892e436f-f840-45dd-abad-032f50270365.jpg)
![Picture13](https://user-images.githubusercontent.com/44596318/142511735-6e738687-4b2f-40b1-8c04-0851433c1678.jpg)
![Picture14](https://user-images.githubusercontent.com/44596318/142511736-262024d6-7f84-427c-8f61-cd984421056f.jpg)
![Picture15](https://user-images.githubusercontent.com/44596318/142511737-37af0ea4-fe8f-4948-a673-b462e06491a5.jpg)
![Picture16](https://user-images.githubusercontent.com/44596318/142511738-9030d710-e18a-4694-bab2-52745545ebb6.jpg)
![Picture17](https://user-images.githubusercontent.com/44596318/142511739-63a8b08f-30c3-44a2-b2dc-3ca553c2225f.jpg)
![Picture18](https://user-images.githubusercontent.com/44596318/142511740-f8bd6716-4e32-485a-944e-a5e1009edceb.jpg)
![Picture19](https://user-images.githubusercontent.com/44596318/142511741-c63425a0-7e1a-4589-9639-ef1571ec0908.jpg)
![Picture20](https://user-images.githubusercontent.com/44596318/142511742-d6eb733f-9faa-474b-ba86-aba79623d098.jpg)
![Picture21](https://user-images.githubusercontent.com/44596318/142511743-19fe2c98-8d79-4716-b22a-e4bfe03a34e3.jpg)
![Picture22](https://user-images.githubusercontent.com/44596318/142511744-05e6c89a-3b05-4219-a4d6-0a7179b4f09d.jpg)

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





