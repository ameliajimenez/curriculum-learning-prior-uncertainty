import os
import numpy as np
from data_loader import read_data_sets
import convnet
from sklearn import metrics

# working directory
workingdir = os.getcwd()
perc = 100
strategy = 'baseline'  # ['baseline', 'reorder', 'subsets', 'weights']'
curriculum_type = 'prior_knowledge'  # ['uncertainty', 'uncertainty']
modeldir = os.path.join('./models/', str(perc), strategy)

my_model = os.path.join(modeldir, 'model'+'.cpkt')

# load data
datadir = os.path.join(os.getcwd(), './data/mnist')
data_provider = read_data_sets(datadir)
x_test = data_provider.test.images  # set of images to evaluate
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')  # pad input
y_test = np.argmax(data_provider.test.labels, 1)  # dense labels

# network definition
net = convnet.ConvNet(channels=1, n_class=10, is_training=False)

# classification performance
n_test = data_provider.test.images.shape[0]
batch_size = 512
predictions = np.zeros_like(y_test)

for count, kk in enumerate(range(0, n_test, batch_size)):
    if count == int(n_test / batch_size):
        start = kk
        end = x_test.shape[0]
    else:
        start = kk
        end = kk + batch_size

    n_samples = end - start
    xxtest = x_test[start:end, ...]

    preds = net.predict(my_model, xxtest)
    predictions[start:end] = np.argmax(np.squeeze(preds), 1)

print('Confusion matrix')
print(metrics.confusion_matrix(y_test, predictions))

print('Metrics report')
print(metrics.classification_report(y_test, predictions))

print('Error Rate')
error_rate = 100.0 - (100.0 * np.sum(predictions == y_test) / (predictions.shape[0]))
print(error_rate)
