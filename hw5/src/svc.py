from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from dataset import dataset

x, y = dataset()
# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Initialize the SVM classifier
svm = SVC(kernel='linear')
svm.fit(x_train, y_train)
accuracy = svm.score(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")