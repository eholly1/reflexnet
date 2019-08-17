import tensorflow as tf


class FeedForward:

	def __init__(
		self,
		input_size,
		output_size,
		layers_config,
		hidden_activation='tanh',
		output_activation='linear',
		):
		self._layers = []
		if layers_config:
			self._layers.append(tf.keras.layers.Dense(layers_config[0], activation=hidden_activation, input_shape=[input_size]))
			for layer_size in layers_config[:1]:
				self._layers.append(tf.keras.layers.Dense(layer_size, activation=hidden_activation))
			self._layers.append(tf.keras.layers.Dense(output_size, activation=output_activation))
		else:
			layers = [tf.keras.layers.Dense(output_size, activation=output_activation, input_shape=[input_size])]

	def __call__(self, x):
		for layer in self._layers:
			x = layer(x)
		return x

