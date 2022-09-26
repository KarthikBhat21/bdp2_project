from flask import Flask
import flask
import pickle
import numpy

app = flask.Flask(__name__)

model = pickle.load(open("SVMmodel.pkl","rb"))

@app.route('/prediction')
def predict():
    features = [ i for i in range(0,500)]
    final_features= [numpy.array(features)]
    prediction = model.predict(final_features)
    return str(prediction)


@app.route('/')
def home_page():
    return "welcome to prediction home"


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')

# from flask import Flask,render_template,url_for,request
# import pandas as pd 
# import numpy as np
# # from nltk.stem.porter import PorterStemmer
# import re
# import string
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression



# app = Flask(__name__)
# model = pickle.load(open("SVMmodel.pkl","rb"))



# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/predict',methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         message = dict(request.form)
#         # message = request.form['message']
#         # data = [message]
#         data = message.values()
#         # data = [list(map(float, data))]
#         response = predict(data)
#         my_prediction = model.predict(data).tolist()[0]
#     return render_template('result.html',prediction = my_prediction)

# # @app.route('/predict',methods=['POST'])
# # def predict():
# #     if request.method == 'POST':
# #         message = request.form['message']
# #         data = [message]
# #         my_prediction = model.predict(data)
# #     return render_template('result.html',prediction = my_prediction)

# # def predict(data):
# #     model = pickle.load(open("SVMmodel.pkl","rb"))

# #     prediction = model.predict(data).tolist()[0]

# #     return prediction 


# # def form_response(dict_request):
# #             data = dict_request.values()
# #             data = [list(map(float, data))]
# #             response = predict(data)
# #             return response

# # @app.route('/predict',methods=['POST'])
# # def index():
# #     if request.method == "POST":
# #         try:
# #             if request.form:
# #                 dict_req = dict(request.form)
# #                 response = form_response(dict_req)
# #                 return render_template("result.html", response=response)
# #         except Exception as e:
# #             print(e)
# #             error = {"error": "Something went wrong!! Try again later!"}
# #             error = {"error": e}
# #             return render_template("404.html", error=error)
# #     else:
# #         return render_template("result.html")


# if __name__ == '__main__':
#     app.run(host='0.0.0.0',port=4000,debug =True)





# from flask import Flask,render_template,url_for,request
# import pandas as pd 
# import numpy as np
# import re
# import string
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
# import os

# app = Flask(__name__)

# ## Definitions
# def remove_pattern(input_txt,pattern):
#     r = re.findall(pattern,input_txt)
#     for i in r:
#         input_txt = re.sub(i,'',input_txt)
#     return input_txt
# def count_punct(text):
#     count = sum([1 for char in text if char in string.punctuation])
#     return round(count/(len(text) - text.count(" ")),3)*100



# input = "data/features"

# data_input = os.path.join(input,"features.csv")

# data = pd.read_csv(data_input)

# data['body_len'] = data['text'].apply(lambda x:len(x) - x.count(" "))
# data['punct%'] = data['text'].apply(lambda x:count_punct(x))
# X = data['text']
# y = data['sentiment']
# # Extract Feature With CountVectorizer
# cv = CountVectorizer()
# X = cv.fit_transform(X) # Fit the Data
# X = pd.concat([data['body_len'],data['punct%'],pd.DataFrame(X.toarray())],axis = 1)
# from sklearn.model_selection import train_test_split
# #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# ## Using Classifier
# clf = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
#                    intercept_scaling=1, l1_ratio=None, max_iter=100,
#                    multi_class='auto', n_jobs=None, penalty='l2',
#                    random_state=None, solver='liblinear', tol=0.0001, verbose=0,
#                    warm_start=False)
# clf.fit(X,y)

# model = pickle.load(open("Logisticmodel.pkl","rb"))


# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/predict',methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         features = [ i for i in range(0,500)]
#         final_features= [numpy.array(features)]
#         message = request.form['message']
#         data = [message]
#         # X = dataset.iloc[:, :500].values
#         vect = pd.DataFrame(cv.transform(data).toarray())
#         body_len = pd.DataFrame([len(data) - data.count(" ")])
#         # body_len = pd.DataFrame.iloc[:, :500].values
#         punct = pd.DataFrame([count_punct(data)])
#         total_data = pd.concat([body_len,punct,vect],axis = 1)
#         my_prediction = model.predict(total_data)
#     return render_template('result.html',prediction = my_prediction)


# if __name__ == '__main__':
#     app.run(host='0.0.0.0',port=4000)