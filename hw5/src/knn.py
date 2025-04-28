from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from dataset import dataset

x , y = dataset()

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Initialize the KNN classifier
for i in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    accuracy = knn.score(x_test, y_test)
    print(f"Accuracy with {i} neighbors: {accuracy:.2f}")
