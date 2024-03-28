# %%
#%%shell
# Installs the latest dev build of TVM from PyPI, with CUDA enabled. To use this,
# you must request a Google Colab instance with a GPU by going to Runtime ->
# Change runtime type -> Hardware accelerator -> GPU. If you wish to build from
# source, see https://tvm.apache.org/docs/install/from_source.html
#! pip install tlcpack-nightly-cu113 --pre -f https://tlcpack.ai/wheels

# %% [markdown]
# 
# 
# # Quick Start Tutorial for Compiling Deep Learning Models
# **Author**: [Yao Wang](https://github.com/kevinthesun), [Truman Tian](https://github.com/SiNZeRo)
# 
# This example shows how to build a neural network with Relay python frontend and
# generates a runtime library for Nvidia GPU with TVM.
# Notice that you need to build TVM with cuda and llvm enabled.
# 

# %% [markdown]
# ## Overview for Supported Hardware Backend of TVM
# The image below shows hardware backend currently supported by TVM:
# 
# <img src="https://github.com/dmlc/web-data/raw/main/tvm/tutorial/tvm_support_list.png" align="center">
# 
# In this tutorial, we'll choose cuda and llvm as target backends.
# To begin with, let's import Relay and TVM.
# 
# 

# %%
import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_executor
import tvm.testing

# %% [markdown]
# ## Define Neural Network in Relay
# First, let's define a neural network with relay python frontend.
# For simplicity, we'll use pre-defined resnet-18 network in Relay.
# Parameters are initialized with Xavier initializer.
# Relay also supports other model formats such as MXNet, CoreML, ONNX and
# Tensorflow.
# 
# In this tutorial, we assume we will do inference on our device and
# the batch size is set to be 1. Input images are RGB color images of
# size 224 * 224. We can call the
# :py:meth:`tvm.relay.expr.TupleWrapper.astext()` to show the network
# structure.
# 
# 

# %%
batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

mod, params = relay.testing.resnet.get_workload(
    num_layers=18, batch_size=batch_size, image_shape=image_shape
)

# set show_meta_data=True if you want to show meta data
print(mod.astext(show_meta_data=False))

# %% [markdown]
# ## Compilation
# Next step is to compile the model using the Relay/TVM pipeline.
# Users can specify the optimization level of the compilation.
# Currently this value can be 0 to 3. The optimization passes include
# operator fusion, pre-computation, layout transformation and so on.
# 
# :py:func:`relay.build` returns three components: the execution graph in
# json format, the TVM module library of compiled functions specifically
# for this graph on the target hardware, and the parameter blobs of
# the model. During the compilation, Relay does the graph-level
# optimization while TVM does the tensor-level optimization, resulting
# in an optimized runtime module for model serving.
# 
# We'll first compile for Nvidia GPU. Behind the scene, :py:func:`relay.build`
# first does a number of graph-level optimizations, e.g. pruning, fusing, etc.,
# then registers the operators (i.e. the nodes of the optimized graphs) to
# TVM implementations to generate a `tvm.module`.
# To generate the module library, TVM will first transfer the high level IR
# into the lower intrinsic IR of the specified target backend, which is CUDA
# in this example. Then the machine code will be generated as the module library.
# 
# 

# %%
opt_level = 3
target = tvm.target.cuda(model='gtx1650', arch="sm_75")
print(f'target arch: {target.arch}')
with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target, params=params)

# %% [markdown]
# ## Run the generate library
# Now we can create graph executor and run the module on Nvidia GPU.
# 
# 

# %%
# create random input
dev = tvm.cuda()
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
# create module
module = graph_executor.GraphModule(lib["default"](dev))
# set input and parameters
module.set_input("data", data)
# run
module.run()
# get output
out = module.get_output(0, tvm.nd.empty(out_shape)).numpy()

# Print first 10 elements of output
print(out.flatten()[0:10])

# %% [markdown]
# ## Save and Load Compiled Module
# We can also save the graph, lib and parameters into files and load them
# back in deploy environment.
# 
# 

# %%
# save the graph, lib and params into separate files
from tvm.contrib import utils

temp = utils.tempdir()
path_lib = temp.relpath("deploy_lib.tar")
lib.export_library(path_lib)
print(temp.listdir())

# %%
# load the module back.
loaded_lib = tvm.runtime.load_module(path_lib)
input_data = tvm.nd.array(data)

module = graph_executor.GraphModule(loaded_lib["default"](dev))
module.run(data=input_data)
out_deploy = module.get_output(0).numpy()

# Print first 10 elements of output
print(out_deploy.flatten()[0:10])

# check whether the output from deployed module is consistent with original one
tvm.testing.assert_allclose(out_deploy, out, atol=1e-5)


