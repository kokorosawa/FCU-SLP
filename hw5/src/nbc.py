import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from dataset import dataset
x, y = dataset()
print(x.shape)
print(y.shape)
# Convert y to 1D array if it is one-hot encoded
# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Initialize the Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(x_train, y_train)
accuracy = gnb.score(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
