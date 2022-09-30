from keras.datasets import mnist    # Import the dataset
from matplotlib import pyplot       # Import the Plotter

# load the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Show dataset shape
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

# Plot the dataset
for i in range(9):  
    pyplot.figure(figsize=(28, 28))
    pyplot.subplot(331 + i)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
    pyplot.show()
