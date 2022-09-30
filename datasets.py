import numpy as np
from tensorflow.keras.datasets import mnist

# Load the Kaggle A-Z dataset
def load_az_dataset(datasetPath):
	# initialize the list of data and labels
	data = []
	labels = []

	# loop over the rows of the A-Z handwritten digit dataset
	for row in open(datasetPath):
		# parse the label and image from the row
		row = row.split(",")
		label = int(row[0])
		image = np.array([int(x) for x in row[1:]], dtype="uint8")

		# images are represented as single channel (grayscale) images
		# that are 28x28=784 pixels -- we need to take this flattened
		# 784-d list of numbers and repshape them into a 28x28 matrix
		image = image.reshape((28, 28))

		# update the list of data and labels
		data.append(image)
		labels.append(label)

    # convert the data and labels to NumPy arrays
	data = np.array(data, dtype="float32")
	labels = np.array(labels, dtype="int")

	# return a 2-tuple of the A-Z data and labels
	return (data, labels)

# Load the MNIST 0-9 dataset
def load_mnist_dataset():
	# load the MNIST dataset and stack the training data and testing
	# data together (we'll create our own training and testing splits
	# later in the project)
	((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
	data = np.vstack([trainData, testData])
	labels = np.hstack([trainLabels, testLabels])
    
	# return a 2-tuple of the MNIST data and labels
	return (data, labels)



def main():
    (dataAZ, lablesAZ) = load_az_dataset("KaggleAZ/Kaggle_AZ_Data.csv")
    print("dataAZ:", dataAZ.shape)
    print("lablesAZ:", lablesAZ.shape)

    (data09, lables09) = load_mnist_dataset()
    print("data09:", data09.shape)
    print("lables09:", lables09.shape)

if __name__ == "__main__":
    main()