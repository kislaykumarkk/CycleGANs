from load_dataset import *
from architecture import *
import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt 
import numpy as np

image_width = 32
image_height = 32
channels_mnist = 1
channels_svhn = 3
learning_rate = 0.01
n_epochs = 20
batch_size = 64

file_name = "train_model.ckpy"

svhn_train_x,svhn_train_y,svhn_test_x,svhn_test_y = data_svhn()
#print(svhn_train_x[1].shape)
mnist_train_x, mnist_train_y, mnist_test_x, mnist_test_y = data_mnist()




image_A = tf.placeholder(tf.float32, [None, image_height, image_width, channels_mnist])
image_B = tf.placeholder(tf.float32, [None, image_height, image_width, channels_svhn])

gen_B_A = generator_convolution(image_A, channels_svhn)
gen_A_B = generator_convolution(image_B, channels_mnist)

dec_A = discriminator_convolution(image_A)
dec_B = discriminator_convolution(image_B)

dec_gen_B_A = discriminator_convolution(gen_B_A)
dec_gen_A_B = discriminator_convolution(gen_A_B)

cycle_A = generator_convolution(gen_B_A, channels_mnist)
cycle_B = generator_convolution(gen_A_B, channels_svhn)


cycle_loss = tf.reduce_mean(tf.abs(image_A-cycle_A)) + tf.reduce_mean(tf.abs(image_B-cycle_B))

dec_loss_A_1 = tf.reduce_mean(tf.squared_difference(dec_A,1))
dec_loss_B_1 = tf.reduce_mean(tf.squared_difference(dec_B,1))

dec_loss_A_2 = tf.reduce_mean(tf.square(dec_gen_B_A))
dec_loss_B_2 = tf.reduce_mean(tf.square(dec_gen_A_B))


dec_loss_A = (dec_loss_B_1 + dec_loss_A_2)/2
dec_loss_B = (dec_loss_B_1 + dec_loss_B_2)/2

gen_loss_B_1 = tf.reduce_mean(tf.squared_difference(dec_gen_B_A,1))
gen_loss_A_1 = tf.reduce_mean(tf.squared_difference(dec_gen_A_B,1))

gen_loss_A = gen_loss_A_1 + 10*cycle_loss
gen_loss_B = gen_loss_B_1 + 10*cycle_loss

optimizer = tf.train.AdamOptimizer(learning_rate)

train_dec_A = optimizer.minimize(dec_loss_A)
train_dec_B = optimizer.minimize(dec_loss_B)
train_gen_A = optimizer.minimize(gen_loss_A)
train_gen_B = optimizer.minimize(gen_loss_B)


saver = tf.train.Saver()
with tf.Session() as sess:
	if os.path.isfile("./"+file_name):
		saver.restore(sess,save_file)
	else:
		sess.run(tf.global_variables_initializer())
	for epoch in range(n_epochs):
		batch_index = 0
		for i in range(int(svhn_train_x.shape[0]/batch_size)):
			print("Epoch: {}, Batch Number: {}".format(epoch,i))
			#x = mnist_train_x[i].reshape(1,mnist_train_x[i].shape[0],mnist_train_x.shape[1],1)
			#print(svhn_train_x[i].shape)
			#y = svhn_train_x[i].reshape(1,svhn_train_x[i].shape[0],svhn_train_x[i].shape[1],svhn_train_x[i].shape[2])

			x = mnist_train_x[batch_index:(batch_index+batch_size)].reshape(64,32,32,1)
			y = svhn_train_x[batch_index:(batch_index+batch_size)]
			
			_, fake_B = sess.run([train_gen_A, gen_B_A], feed_dict={image_A:x, image_B:y})

			_ = sess.run([train_dec_B], feed_dict={image_A:x,image_B:y})

			_, fake_A = sess.run([train_gen_B, gen_A_B], feed_dict={image_A:x,image_B:y})

			_ = sess.run([train_dec_A], feed_dict={image_A:x,image_B:y})
			
		loss1_list = []
		loss2_list = []
		for i in range(int(svhn_train_x.shape[0]/batch_size)):
			#train_x = mnist_train_x.reshape(mnist_train_x.shape[0],mnist_train_x.shape[1],mnist_train_x.shape[2],1)
			x_new = mnist_train_x[batch_index:(batch_index+batch_size)].reshape(64,32,32,1)
			y_new = svhn_train_x[batch_index:(batch_index+batch_size)]
			loss1, loss2 = sess.run([gen_loss_A,gen_loss_B], feed_dict={image_A:x_new, image_B:y_new})
			loss1_list.append(loss1)
			loss2_list.append(loss2)
			#gen_loss_B = sess.run(gen_loss_B, feed_dict={image_A:train_x, image_B:svhn_train_x})
		print("Generator A Loss:",sum(loss1_list)/len(loss1_list))
		print("Generator B Loss:",sum(loss2_list)/len(loss2_list))

		batch_index = batch_index+batch_size
	saver.save(sess,file_name)
	print("Trained Model Saved")














