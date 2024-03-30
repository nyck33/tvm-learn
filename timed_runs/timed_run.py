import tvm
from tvm import relay
import time

# Load the exported ONNX model
onnx_model = onnx.load('matmul.onnx')

# Convert the ONNX model to Relay IR
mod, params = relay.frontend.from_onnx(onnx_model, {'input': dummy_input.shape})

# Set the target to 'cuda' for GPU
target = 'cuda'

# Build the Relay IR to executable with TVM
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build_module.build(mod, target, params=params)

# Create TVM runtime and load the module
ctx = tvm.gpu(0)
module = graph_runtime.create(graph, lib, ctx)

# Set input and parameters
module.set_input('input', tvm.nd.array(dummy_input.numpy()))
module.set_input(**params)

# Run the module
start_time = time.time()
module.run()
end_time = time.time()

# Get the output
tvm_output = module.get_output(0)

# Calculate and print the time taken
execution_time = end_time - start_time
print(f"Execution Time: {execution_time} seconds")
