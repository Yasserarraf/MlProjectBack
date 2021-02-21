# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import re
# import string
#
# url1 = "C:/Users/DELL/Desktop/ML/Projet/Fake.csv"
# url2 = "C:/Users/DELL/Desktop/ML/Projet/True.csv"
#
# df_fake = pd.read_csv(url1)
# df_true = pd.read_csv(url2)
#
#
# print(df_fake.head(10))
# print(df_true.head(10))
#
# #classifiy news by a numerical value
# df_fake["class"] = 0
# df_true["class"] = 1
#
#
# print(df_fake.shape)
# print(df_true.shape)
#
#
#
# #save data to manual_testing.csv file
# df_fake_manual_testing = df_fake.tail(10)
# for i in range( 23480 ,23470  ,  -1):
#     df_fake.drop([i], axis=0, inplace=True)
# df_true_manual_testing = df_true.tail(10)
# for i in range( 21416 ,21406  ,  -1):
#     df_true.drop([i], axis=0, inplace=True)
#
# #merge the data to data set in a single data frame
# df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing],axis=0)
# df_manual_testing.to_csv("manual_testing.csv")
#
# #df_m = pd.read_csv('manual_testing.csv')
# #print(df_m)
#
#
# df_marge = pd.concat([df_fake, df_true], axis=0)
# #print(df_marge.head(10))
#
# df = df_marge.drop(["title", "subject", "date"], axis=1)
# #print(df.head(10))
#
# df = df.sample(frac=1)
# #print(df.head(10))
#
# #check is there any null value is present or not
# #print(df.isnull().sum())
#
#
# #remove all the special an unecessary characters
# def word_drop(text):
#     text = text.lower()
#     text = re.sub('\[.*?\]','' , text)
#     text = re.sub("\\W", " " , text)
#     text = re.sub('https?://\S+|www\S+', '', text)
#     text = re.sub('<.*?>+' , '', text)
#     text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
#     text = re.sub('\n', '', text)
#     text = re.sub('\w*\d\w*' , '' , text)
#     return text
#
# df["text"] = df["text"].apply(word_drop)
# #print(df.head(10))
#
# #define dependent and independent variable as x and y
# x = df["text"]
# y = df["class"]
#
# #splitting the data set into train and test
# x_train, x_test, y_train, y_test = train_test_split(x,y ,test_size= .25)
#
# #vectorize x
# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorization = TfidfVectorizer()
# xv_train = vectorization.fit_transform(x_train)
# xv_test = vectorization.transform(x_test)
#
#
# """"--------Classification------------"""
# ##########Logistic Regression###########
# from sklearn.linear_model import LogisticRegression
# LR = LogisticRegression()
# print(LR.fit(xv_train, y_train))
#
# print(LR.score(xv_test, y_test))
#
# #classification report
# pred_LR = LR.predict(xv_test)
# print(classification_report(y_test, pred_LR))
#
#
#
# ##########Decision Tree Classification###########
# from sklearn.tree import DecisionTreeClassifier
# DT = DecisionTreeClassifier()
# print(DT.fit(xv_train, y_train))
#
# print(DT.score(xv_test, y_test))
#
# #classification predict
# pred_DT = DT.predict(xv_test)
# print(classification_report(y_test, pred_DT))
#
#
# #########Gradient Boosting Classifier###########
# from sklearn.ensemble import GradientBoostingClassifier
# GBC = GradientBoostingClassifier(random_state=0)
# print(GBC.fit(xv_train, y_train))
#
# print(GBC.score(xv_test, y_test))
#
# #classification predict
# pred_GBC = GBC.predict(xv_test)
# print(classification_report(y_test, pred_GBC))
#
#
# #########Random Forest Classifier#######
# from sklearn.ensemble import RandomForestClassifier
# RFC = RandomForestClassifier(random_state=0)
# print(RFC.fit(xv_train, y_train))
#
# print(RFC.score(xv_test, y_test))
#
# #classification predict
# pred_RFC = RFC.predict(xv_test)
# print(classification_report(y_test, pred_RFC))
#
#
# """------------Manual Testing------"""
# def output_lable(n):
#         if n == 0:
#             return "Fake News"
#         elif n == 1:
#             return "Not A Fake News"
#
#
# def manual_testing(news):
#     testing_news = {"text":[news]}
#     new_def_test = pd.DataFrame(testing_news)
#     new_def_test["text"] = new_def_test["text"].apply(word_drop)
#     new_x_test = new_def_test["text"]
#     new_xv_test = vectorization.transform(new_x_test)
#     pred_LR = LR.predict(new_xv_test)
#     pred_DT = DT.predict(new_xv_test)
#     pred_GBC = GBC.predict(new_xv_test)
#     pred_RFC = RFC.predict(new_xv_test)
#
#     return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]),
#                                                                                                               output_lable(pred_DT[0]),
#                                                                                                               output_lable(pred_GBC[0]),
#                                                                                                               output_lable(pred_RFC[0])))
#
# news = str(input())
# print(manual_testing(news))
