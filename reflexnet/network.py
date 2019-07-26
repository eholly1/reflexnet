import torch

def _module_list_forward(module_input, module_list):
	x = module_input
	for m in module_list:
		x = m(x)
	return x

class FeedForward(torch.nn.Module):

	def __init__(self, input_size, output_size, layers_config, output_activation=None):
		super().__init__()
		modules = []
		last_size = input_size
		for layer_size in layers_config:
			modules.append(torch.nn.Linear(last_size, layer_size))
			modules.append(torch.nn.ReLU())
			last_size = layer_size
		modules.append(torch.nn.Linear(last_size, output_size))

		if output_activation is not None:
			modules.append(output_activation)

		self._module_list = torch.nn.ModuleList(modules)

	def forward(self, x):
		return _module_list_forward(x, self._module_list)

class SoftKNN(torch.nn.Module):

	@property
	def k(self):
		return self._k

	def set_point(self, input_value, output_value):
		if self._i > self._k:
			raise ValueError('Cannot set anymore points.')
		self._mean[i] = input_value
		self._outputs[i] = output_value
		self._i += 1

	def __init__(
		self,
		input_size,
		output_size,
		init_stdev=0.1, 
		k=200):
		super().__init__()
		self._k = k
		self._i = 0
		self._mean = torch.nn.parameter.Parameter(
			torch.zeros(k, input_size))
		self._stddev = torch.nn.parameter.Parameter(
			torch.ones(k, input_size) * init_stdev)
		self._distribution = torch.distributions.normal.Normal(
			loc=self._mean,
			scale=self._stddev,
		)
		self._softmax = torch.nn.Softmax(dim=-1)  # Softmax dim should be across k.
		self._outputs = torch.nn.parameter.Parameter(
			torch.zeros(k, output_size))

	def initialize_points(self, input_mean, input_stddev, output_mean, output_stddev):
		input_init_dist = torch.distributions.normal.Normal(input_mean, input_stddev)
		input_init_sample = input_init_dist.sample(sample_shape=[self._k])
		self._mean.data = input_init_sample.data

		output_init_dist = torch.distributions.normal.Normal(output_mean, output_stddev)
		output_init_sample = output_init_dist.sample(sample_shape=[self._k])
		self._outputs.data = output_init_sample.data

	def forward(self, x):
		if len(x.shape) > 1:
			# If there is a batch dimension, add a dimension to broadcast with
			#   reflex dimension.
			x = torch.unsqueeze(x, dim=1)

		lp = self._distribution.log_prob(x)
		joint_lp = torch.sum(lp, dim=-1)  # Sum log probs over obs dim.
		softmax_weights = self._softmax(joint_lp)
		weighted_outputs = self._outputs * softmax_weights.unsqueeze(dim=-1)
		outputs = weighted_outputs.sum(dim=-2)  # Sum over k.
		return outputs
