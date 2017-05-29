from settings import *
import model
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
sns.set()

dataset = pd.read_csv(location_data)
Y = dataset[[inference_column]].values
label = np.unique(Y).tolist()
try:
    Y = LabelEncoder().fit_transform(Y)
except:
    print 'Y already in integer value'
    
onehot_label = np.zeros((Y.shape[0], len(label)), dtype = np.float32)
for i in xrange(Y.shape[0]):
    onehot_label[i][Y[i]] = 1.0
    
X = dataset[input_columns].values
X = StandardScaler().fit_transform(X)
X = Normalizer().fit_transform(X)

X_components = PCA(n_components = 2).fit_transform(X)

model = model.Model(num_layers, size_layers, enable_dropout, dropout_probability, enable_penalty, penalty, activation_functions, last_activation_function, 
                    loss_function, learning_rate, X_components.shape[1], onehot_label.shape[1], optimizer)


X_train, X_test, Y_train, Y_test, label_train, label_test = train_test_split(X_components, onehot_label, Y, test_size = split_test)
EPOCH = []; LOSS = []; ACCURACY = [];

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in xrange(epoch):
    
    EPOCH.append(i)
    last_time = time.time()
    total_accuracy = 0
    total_loss = 0
    
    for n in xrange(0, X_train.shape[0] - batch_size, batch_size):
        
        _, loss = sess.run([model.optimizer, model.cost], feed_dict = {model.X: X_train[n: n + batch_size, :], model.Y: Y_train[n: n + batch_size, :]})
        acc = sess.run(model.accuracy, feed_dict = {model.X: X_train[n: n + batch_size, :], model.Y: Y_train[n: n + batch_size, :]})
        
        total_accuracy += acc
        total_loss += loss
        
    total_accuracy = total_accuracy / ((X_train.shape[0] - batch_size) / (batch_size * 1.0))
    total_loss = total_loss / ((X_train.shape[0] - batch_size) / (batch_size * 1.0))
    
    LOSS.append(total_loss)
    ACCURACY.append(total_accuracy)
    
    if (i + 1) % checkpoint == 0:
        print 'iteration: ' + str(i + 1) + ' accuracy: ' + str(total_accuracy) + ' loss: ' + str(total_loss) + ' seconds per iteration: ' + str(time.time() - last_time)

test_accuracy = sess.run(model.accuracy, feed_dict = {model.X: X_test, model.Y: Y_test})

plt.figure(figsize = (10, 5))
plt.subplot(1, 2, 1)
plt.plot(EPOCH, LOSS)
plt.title('Loss graph')
plt.subplot(1, 2, 2)
plt.plot(EPOCH, ACCURACY)
plt.title('Accuracy graph')
plt.savefig('training.pdf')
plt.cla()

plt.figure(figsize = (30, 10))
x_min, x_max = X_components[:, 0].min() - 0.5, X_components[:, 0].max() + 0.5
y_min, y_max = X_components[:, 1].min() - 0.5, X_components[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))
ax = plt.subplot(1, 2, 1)
ax.set_title('Input data')
ax.scatter(X_train[:, 0], X_train[:, 1], c = label_train, cmap = plt.cm.Set1, label = label)
ax.scatter(X_test[:, 0], X_test[:, 1], c = label_test, cmap = plt.cm.Set1, alpha = 0.6)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
ax = plt.subplot(1, 2, 2)
contour = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
Z = sess.run(tf.nn.softmax(model.last_l), feed_dict = {model.X: contour})
temp_answer = []
for q in xrange(Z.shape[0]):
    temp_answer.append(np.argmax(Z[q]))
Z = np.array(temp_answer)
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap = plt.cm.Set1, alpha = 0.8)
ax.scatter(X_train[:, 0], X_train[:, 1], c = label_train, cmap = plt.cm.Set1, label = label)
ax.scatter(X_test[:, 0], X_test[:, 1], c = label_test, cmap = plt.cm.Set1, alpha = 0.6)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('hypothesis space')
ax.text(xx.min() + 0.3, yy.min() + 0.3, str(test_accuracy), size = 15)
plt.tight_layout()
plt.savefig('neuralnet_hypothesis.pdf')

