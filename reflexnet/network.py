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

class LatentGaussianNetwork(torch.nn.Module):

	def __init__(self, input_size, output_size, layers_config, latent_size=10):
		super().__init__()
		self._input_size = input_size
		self._output_size = output_size
		self._latent_size = latent_size
		self._encoder = FeedForward(self._input_size, self._latent_size, layers_config)
		self._decoder = FeedForward(self._latent_size, self._output_size, layers_config)
		self._latent_prior = torch.distributions.normal.Normal(
			loc=torch.tensor([0.0] * self._latent_size),
			scale=torch.tensor([1.0] * self._latent_size),
			)

	def forward(self, x):
		outputs, unused_l2 = self.forward_full(x)
		return outputs

	def forward_full(self, x):
		latent_means = self._encoder(x)
		l2 = latent_means.pow(2).mean(dim=-1)
		latent_sample = self._latent_prior.sample() + latent_means
		outputs = self._decoder(latent_sample)
		return outputs, l2

	def latent_l2(self, x):
		latent_means = self._encoder(x)
		l2 = latent_means.pow(2).sum(dim=-1)
		return l2
