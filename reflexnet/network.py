import torch

import summaries

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
		with torch.no_grad():
			idx = self._least_used_reflex()
			self._mean[idx] = input_value

			# Initialize stddev to mean stddev.
			self._stddev[idx] = torch.mean(self._stddev, dim=0)
			
			self._outputs[idx] = output_value
			self._top_k_counts[idx] = 1.0
			self._total_counts[idx] = 1.0

	def __init__(
		self,
		input_size,
		output_size,
		init_stdev=1.0,
		top_k=10,
		k=1000):
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
		self._top_k = top_k
		self.global_step = 0

		# Count of the number of times each reflex has been used.
		self._top_k_counts = torch.nn.parameter.Parameter(torch.zeros(k), requires_grad=False)

		# Count of the number of opportunities each reflex has been involved in.
		self._total_counts = torch.nn.parameter.Parameter(torch.zeros(k), requires_grad=False)

	def _least_used_reflex(self):
		usages = self._top_k_counts / self._total_counts
		_, idx = torch.topk(-usages, 1)
		return idx

	def initialize_points(self, input_mean, input_stddev, output_mean, output_stddev):
		input_init_dist = torch.distributions.normal.Normal(input_mean, input_stddev)
		input_init_sample = input_init_dist.sample(sample_shape=[self._k])
		self._mean.data = input_init_sample.data

		output_init_dist = torch.distributions.normal.Normal(output_mean, output_stddev)
		output_init_sample = output_init_dist.sample(sample_shape=[self._k])
		self._outputs.data = output_init_sample.data

	def softmax_weights(self, x):
		"""Get the softmax weights across reflexes, for an input tensor.
		If top_k is not None, returns only the top K softmax weights, with the indices
		for those reflexes.
		"""
		if len(x.shape) > 1:
			# If there is a batch dimension, add a dimension to broadcast with
			#   reflex dimension.
			x = torch.unsqueeze(x, dim=1)

		lp = self._distribution.log_prob(x)
		joint_lp = torch.sum(lp, dim=-1)  # Sum log probs over obs dim.

		# If top_k is not None, take only topk log_probabilities.
		softmax_topk_idxs = None
		if self._top_k is not None:
			joint_lp, softmax_topk_idxs = torch.topk(
				joint_lp, self._top_k, dim=-1)
		
		softmax_weights = self._softmax(joint_lp)

		return softmax_weights, softmax_topk_idxs

	def forward(self, x):
		softmax_weights, softmax_topk_idxs = self.softmax_weights(x)

		if self.training:
			summaries.add_histogram(
				"softmax_weights", softmax_weights, self.global_step)

		if softmax_topk_idxs is None:
			outputs = self._outputs
		else:
			outputs = self._outputs[softmax_topk_idxs, :]
		
		self._top_k_counts[softmax_topk_idxs] += 1.0
		self._total_counts += 1.0

		weighted_outputs = outputs * softmax_weights.unsqueeze(dim=-1)
		outputs = weighted_outputs.sum(dim=-2)  # Sum over k.
		return outputs
