import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Read csv of emails determining whether they are spam or ham
df = pd.read_csv("mail_data.csv") 
data = df.where((pd.notnull(df)), '')

# Have spam emails be resembled by a 0 and ham emails as 1
data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1

# Set x to messages and y to categories
x = data['Message']
y = data['Category'].astype('int')

# Train 80% of data and test 20% x and y data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state = 3)

# Transform text data to vectors that can be evaluated
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True) # Transfer to feature vector

x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

y_train_features = y_train.astype('int')
y_test_features = y_test.astype('int')

model = LogisticRegression()
model.fit(x_train_features, y_train)

# Prediction for model based on training data
prediction_on_training_data = model.predict(x_train_features)
accuracy_on_training_data = accuracy_score(y_train, prediction_on_training_data)

# Prediction for model based on test data
prediction_on_test_data = model.predict(x_test_features)
accuracy_on_test_data = accuracy_score(y_test, prediction_on_test_data) 


# Get input email from text file
with open('email.txt', 'r') as file:
    input_email = file.read().rstrip()
formatted_input_email = [input_email]
input_data_features = feature_extraction.transform(formatted_input_email)

prediction = model.predict(input_data_features)
probability = model.predict_proba(input_data_features)

if(prediction[0] == 1):
    print("Ham mail")
    print(f"Confidence: {probability[0][1] * 100:.2f}%")
else:
    print("Spam mail")
    print(f"Confidence: {probability[0][0] * 100:.2f}%")
