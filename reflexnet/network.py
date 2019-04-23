import torch

def _module_list_forward(module_input, module_list):
	x = module_input
	for m in module_list:
		x = m(x)
	return x

class FeedForward(torch.nn.Module):

	def __init__(self, input_size, output_size, layers_config, output_activation=torch.nn.Identity):
		super().__init__()
		modules = []
		last_size = input_size
		for layer_size in layers_config:
			modules.append(torch.nn.Linear(last_size, layer_size))
			modules.append(torch.nn.ReLU())
			last_size = layer_size
		modules.append(torch.nn.Linear(last_size, output_size))
		modules.append(torch.nn.Identity())
		self._module_list = torch.nn.ModuleList(modules)

	def forward(self, x):
		return _module_list_forward(x, self._module_list)
