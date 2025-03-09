from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from loguru import logger

from preprocess import read_wavefile

logger.add('logs/main.log')

logger.info('loading data...')
x, y = read_wavefile('wavefiles-all')
# print(x.shape, y.shape)
# print(x[:10])
# print(y[:10])

logger.info('splitting data...')
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

model = GaussianNB()
logger.info('training model...')
model.fit(train_x, train_y)

pred_y = model.predict(test_x)
logger.info('evaluating model...')
# print(pred_y[:10])
# print(test_y[:10])
accuracy = accuracy_score(test_y, pred_y)

print('Accuracy: ', accuracy*100, '%')