import time
import torch

import network

N = 10000
batch_size = 64
input_size = 20
output_size = 10
ff = network.FeedForward(input_size, output_size, [300])

class SubnetModule(torch.jit.ScriptModule):

    __constants__ = ['_subnets']

    def __init__(self):
        super().__init__()
        subnets = []
        for _ in range(60):
            subnets.append(
            # subnets.append(torch.jit.trace(
                network.FeedForward(input_size, output_size, [5]),
                # torch.rand(batch_size, input_size)
            # ))
            )
        self._subnets = torch.nn.ModuleList(subnets)

    @torch.jit.script_method
    def forward(self, x):
        output = torch.zeros(10)
        for s in self._subnets:
            output += torch.sum(s(x), dim=0)
        return output

subnet = SubnetModule()

time_ff = 0.0
time_subnets = 0.0

print("Running %d samples." % N)
for i in range(N):
    inputs = torch.rand(batch_size, input_size)

    start = time.time()
    outputs = ff(inputs)
    time_ff += time.time() - start

    start = time.time()
    outputs = subnet(inputs)
    time_subnets += time.time() - start

print("FF Time: %f" % time_ff)
print("Subnets Time: %f" % time_subnets)
time_ff = 0.0
time_subnets = 0.0

traced_ff = torch.jit.trace(ff, inputs)
traced_subnet = torch.jit.trace(subnet, inputs)

print("Running %d samples." % N)
for i in range(N):
    inputs = torch.rand(batch_size, input_size)

    start = time.time()
    outputs = traced_subnet(inputs)
    time_ff += time.time() - start

    start = time.time()
    outputs = traced_subnet(inputs)
    time_subnets += time.time() - start

print("Traced FF Time: %f" % time_ff)
print("Traced Subnets Time: %f" % time_subnets)
time_ff = 0.0
time_subnets = 0.0

print("Running %d samples." % N)
for i in range(N):
    inputs = torch.rand(batch_size, input_size)

    start = time.time()
    outputs = traced_subnet(inputs)
    time_ff += time.time() - start

    start = time.time()
    outputs = traced_subnet(inputs)
    time_subnets += time.time() - start

print("Traced FF Time: %f" % time_ff)
print("Traced Subnets Time: %f" % time_subnets)
time_ff = 0.0
time_subnets = 0.0

