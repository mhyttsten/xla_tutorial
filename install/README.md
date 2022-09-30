# Installation Instructions

## Prerequisites
In general, you should install all components needed to build TensorFlow from source. Probably, you'll be fine installing the following components:
- C++ compiler toolchain
- git
- bazel

But please refer to the full documentation (https://www.tensorflow.org/install/source) for completeness.

## Install TensorFlow source and checkout the 2.10 version
- git clone https://github.com/tensorflow/tensorflow.git
- cd tensorflow
- git checkout r2.10
- cd ..

## Install xla_tutorial
- git clone https://github.com/mhyttsten/xla_tutorial.git
- cp xla_tutorial/install/BUILD_tensorlow_tensorflow_core_profiler_backends_cpu tensorflow/tensorflow/core/profiler/backends/cpu/BUILD
- cp xla_tutorial/install/BUILD_tensorlow_tensorflow_core_profiler_utils tensorflow/tensorflow/core/profiler/utils/BUILD
- cp -R xla_tutorial tensorflow/tensorflow/compiler

## Building and test running a tutorial
- cd tensorflow
- bazel build //tensorflow/compiler/xla_tutorial/01_FirstHLOProgram:01_first_hlo_program

The last instruction runs the cc_binary build rule 01_first_hlo_program in the tensorflow/compiler/xla_tutorial/01_FirstHLOProgram/BUILD file. This rule compiles the 01_first_hlo_program.cc file into an executable, and it will take time. Testing on a 2019 MacBook Pro it took 30min, but depending on your system it might take more (more the 7300+ rules must finish, see the progress in the bazel build output).

Now you should be able to execute the first tutorial. You don't need to understand what the program does or the output, but you should see a successfull completion:
- bazel-bin/tensorflow/compiler/xla_tutorial/01_first_hlo_program/01_first_hlo_program

Hopefully everything has gone well to this point.
If so, you are now all setup and ready to learn everything about how XLA works!

