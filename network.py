import tensorflow as tf

def convolution(input, num_outputs, k_size, variable_name="convolution", activation=tf.nn.relu):
	#with tf.variable_scope("convolution"):
	conv = tf.contrib.layers.conv2d(input, num_outputs=num_outputs, kernel_size=k_size, padding='same', activation_fn= activation)
	#pool = tf.contrib.layers.max_pool2d(conv, kernel_size=[2,2], stride=2)
	norm = tf.layers.batch_normalization(conv)

	return norm


def resnet(input, num_outputs, k_size, variable_name="resnet"):
	#with tf.variable_scope("resnet"):
	conv1 = convolution(input, num_outputs, k_size)
	conv2 = convolution(conv1, num_outputs, k_size)
	resnet_output = conv2 + input

	return resnet_output


def deconvolution(input, num_outputs, k_size, variable_name="deconvolution"):
	#with tf.variable_scope("deconvolution"):
	deconv = tf.contrib.layers.conv2d_transpose(input, num_outputs=num_outputs, kernel_size=k_size, padding='same', activation_fn=tf.nn.relu)
	norm = tf.layers.batch_normalization(deconv)

	return norm