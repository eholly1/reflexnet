import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence

import summaries

def _module_list_forward(module_input, module_list):
	x = module_input
	for m in module_list:
		x = m(x)
	return x

class FeedForward(torch.nn.Module):

	@property
	def sequential(self):
		return False

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

class HRMLayer(torch.nn.Module):

	def __init__(
		self,
		input_size,
		output_size,
		time_delay,
		layers_config,
		):
		super().__init__()
		self._output_size = output_size
		self._network = FeedForward(
			input_size=input_size,
			output_size=self._output_size,
			output_activation=torch.nn.Sigmoid(),
			layers_config=layers_config,
		)
		self._time_delay = time_delay
		self.reset()

	def reset(self):
		self._last_output = torch.zeros(1, self._output_size)

	def forward(self, x, counter):
		if counter % self._time_delay == 0:
			self._last_output = self._network(x)
		return self._last_output
		

class HRMNet(torch.nn.Module):
	"""Hierarchical Reflex Modulation (HRM) Network"""

	@property
	def sequential(self):
		return True

	def reset(self):
		self._counter = -1
		for layer in self._layers:
			if isinstance(layer, HRMLayer):
				layer.reset()

	def __init__(
		self,
		input_size,
		output_size,
		output_reflex_factor=5,
		layers_config=[16],
		latent_time_delays=[3, 10, 30],
		latent_control_dims=[64, 32, 16],
		):
		super().__init__()
		self._time_delays = latent_time_delays

		self._layers = []

		# Output Layer (MLP)
		self._output_size = output_size
		self._output_reflex_factor = output_reflex_factor
		output_layer_dim = output_size * self._output_reflex_factor
		self._layers.append(FeedForward(
			input_size=input_size,
			output_size=output_size * output_reflex_factor,
			layers_config=layers_config,
			output_activation=torch.nn.Tanh()
		))

		# Upper Layers (MLPs)
		for time_delay, control_dim in zip(latent_time_delays, latent_control_dims):
			self._layers.append(HRMLayer(
				input_size=input_size,
				output_size=control_dim,
				time_delay=time_delay,
				layers_config=layers_config,
			))

		self._layers = torch.nn.ModuleList(self._layers)
		
		# Inter-layer Modulation Params
		self._modulation_params = []
		control_dims = [output_layer_dim] + latent_control_dims
		for i in range(len(control_dims) - 1):
			self._modulation_params.append(torch.nn.Parameter(
				torch.ones(control_dims[i+1], control_dims[i])))
		self._modulation_activation_fns = [torch.nn.Sigmoid() for _ in self._modulation_params]

		self.reset()

	def _forward_packed_sequence(self, x):
		outputs = []
		idx = 0

		# For each frame, take the batch and run single step on it.
		for counter, batch_size in enumerate(x.batch_sizes):
			x_step = x[idx:idx+batch_size]
			outputs.append(self._forward_single_frame(x_step, counter=counter))
			idx += batch_size

		# Put together outputs in new PackedSequence.
		return PackedSequence(
			data=torch.cat(outputs, dim=0),
			batch_sizes=x.batch_sizes)

	def _forward_single_frame(self, x, counter):
		# Outputs of each layer of hierarchy.
		layer_outputs = []
		for layer in self._layers:
			if isinstance(layer, HRMLayer):
				layer_outputs.append(layer(x, counter))
			else:
				layer_outputs.append(layer(x))

		# Tuples of modulation matrix params, and their activation functions.
		modulation = list(zip(self._modulation_params, self._modulation_activation_fns))
		
		# Start with output from top layer.
		output = layer_outputs.pop(-1)

		# For each subsequent layer, compute modulation vector and modulate its
		#   output.
		while len(layer_outputs) > 0:
			mod_params, mod_act_fn = modulation.pop(-1)
			next_layer_output = layer_outputs.pop(-1)
			mod_weights = mod_act_fn(mod_params)
			if len(output.shape) == 1:
				output = torch.unsqueeze(output, dim=0)
			mod_vals = torch.mm(output, mod_weights)
			output = next_layer_output * mod_vals
 
		# Final layer has redundant outputs for each output dim. Weighted sum over
		#   each output dim.
		output = output.view(-1, self._output_size, self._output_reflex_factor)
		mod_vals = mod_vals.view(-1, self._output_size, self._output_reflex_factor)
		output = torch.sum(output, dim=-1) / torch.sum(mod_vals, dim=-1)

		return output

	def forward(self, x):
		if isinstance(x, PackedSequence):
			return self._forward_packed_sequence(x)
		self._counter += 1
		output = self._forward_single_frame(x, self._counter)
		if len(x.shape) == 1:
			output = torch.squeeze(output)
		return output

		

class SoftKNN(torch.nn.Module):

	@property
	def sequential(self):
		return False

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
