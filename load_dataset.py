import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt 
import scipy


train_data_size = 1800
test_data_size = 900


def one_hot(Y, num_classes):
    Y_one_hot = np.zeros((Y.shape[0], num_classes))
    for i in range(Y.shape[0]):
    	#print(Y[i][0])
        Y_one_hot[i, Y[i][0]] = 1
    return Y_one_hot


def data_svhn():
	train_mat = scipy.io.loadmat('train_32x32.mat')
	test_mat = scipy.io.loadmat('test_32x32.mat')

	x_train = train_mat['X']
	x_train = np.asarray(x_train)
	y_train = train_mat['y']
	y_train = np.asarray(y_train)

	x_test = test_mat['X']
	x_test = np.asarray(x_test)
	y_test = test_mat['y']
	y_test = np.asarray(y_test)


	x_train_clip = []
	y_train_clip = []
	x_test_clip = []
	y_test_clip = []


	counter_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
	counter_test = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}


	i = 0

	while(sum(counter_train.values())!=train_data_size):
		digit = y_train[i,0]
		for j in range(1,10):
			if digit==j and counter_train[j]<200:
				counter_train[j]+=1
				
				x_train_clip.append(x_train[:,:,:,i]/255)
				y_train_clip.append(y_train[i])

		i+=1

	i = 0

	while(sum(counter_test.values())!=test_data_size):
		digit = y_test[i,0]
		for j in range(1,10):
			if digit==j and counter_test[j]<100:
				counter_test[j]+=1
				
				x_test_clip.append(x_test[:,:,:,i]/255)
				y_test_clip.append(y_test[i])

		i+=1

	x_train_clip = np.asarray(x_train_clip)
	y_train_clip = np.asarray(y_train_clip)
	x_test_clip = np.asarray(x_test_clip)
	y_test_clip = np.asarray(y_test_clip)

	y_train_clip = one_hot(y_train_clip,10)
	y_test_clip = one_hot(y_test_clip,10)


	return x_train_clip,y_train_clip,x_test_clip,y_test_clip


def data_mnist():
	mnist = input_data.read_data_sets(".")

	x_train = mnist.train.images
	y_train = mnist.train.labels
	x_test = mnist.test.images
	y_test = mnist.test.labels

	#return x_train,y_train,x_test,y_test


	x_train_clip = []
	y_train_clip = []
	x_test_clip = []
	y_test_clip = []


	counter_train = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
	counter_test = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}


	i = 0

	while(sum(counter_train.values())!=train_data_size):
		digit = y_train[i]
		for j in range(1,10):
			if digit==j and counter_train[j]<200:
				counter_train[j]+=1
				
				x_train_clip.append(x_train[i,:])
				y_train_clip.append(y_train[i])

		i+=1

	i = 0

	while(sum(counter_test.values())!=test_data_size):
		digit = y_test[i]
		for j in range(1,10):
			if digit==j and counter_test[j]<100:
				counter_test[j]+=1
				
				x_test_clip.append(x_test[i,:])
				y_test_clip.append(y_test[i])

		i+=1

	x_train_clip = np.asarray(x_train_clip)
	x_train_clip = x_train_clip.reshape((x_train_clip.shape[0], 28,28))
	y_train_clip = np.asarray(y_train_clip)
	y_train_clip = y_train_clip.reshape((y_train_clip.shape[0],1))
	x_test_clip = np.asarray(x_test_clip)
	x_test_clip = x_test_clip.reshape((x_test_clip.shape[0], 28,28))
	y_test_clip = np.asarray(y_test_clip)
	y_test_clip = y_test_clip.reshape((y_test_clip.shape[0],1))

	y_train_clip = one_hot(y_train_clip,10)
	y_test_clip = one_hot(y_test_clip,10)


	x_train_resize = np.ndarray((x_train_clip.shape[0],32,32), dtype=float)
	x_test_resize = np.ndarray((x_test_clip.shape[0],32,32), dtype=float)


	for i in range(x_train_clip.shape[0]):
		x_train_resize[i] = cv2.resize(x_train_clip[i], (32, 32),interpolation = cv2.INTER_LINEAR)

	for i in range(x_test_clip.shape[0]):
		x_test_resize[i] = cv2.resize(x_test_clip[i], (32,32),interpolation = cv2.INTER_LINEAR)



	return x_train_resize,y_train_clip,x_test_resize,y_test_clip


