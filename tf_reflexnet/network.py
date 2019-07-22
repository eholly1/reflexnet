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
