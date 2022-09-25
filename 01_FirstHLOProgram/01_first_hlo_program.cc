/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/*
tf_xla/tensorflow/core/profiler//utils/BUILD
tf_xla/tensorflow/core/profiler//lib/BUILD



Optional clean, if desired
  $ bazel clean --expunge

Compile
  $ bazel build --copt="-fno-inline-functions" //tensorflow/compiler/xla_tutorial:t05_first_hlo_program
  (alt use --verbose_failures after 'build' to get verbose build output)
   -fno-inline-functions allow call tracing to work properly for inlined function definitions

Run
  $ bazel-bin/tensorflow/compiler/xla_tutorial/t05_first_hlo_program

INFO: Elapsed time: 1484.648s, Critical Path: 1463.03s


This tutorial creates a program using XLA and HLO
- The program is created using XLS's operation object model in C++.
  In pseudocode, the program is does the following,
  taking int param[3] as argument to its execution:
    const_1 = [[1,2,3], [4,5,6]];
    const_2 = 1
    reduce = xla_Reduce('add', const_1, dim=0, init_values=const_2);
    // reduce = Use initial value:1, then reduce column wise (dim=0), result: [6,8,10]
    result = xla_Map('add', reduce, param);
    // result = [6+param[0], 8+param[1], 10+param[2]];
- The program is compiled
- And finally, the program is executed two times:
  program(param=[10, 11, 12]);  // Result of computation is [16, 19, 22]
  program(param=[11, 12, 13]);  // Result of computation is [17, 20, 23]

==============================================================================*/

#include <iostream>
#include "tensorflow/compiler/xla/client/executable_build_options.h"  // ExecutableBuildOptions
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"  // StreamExecutorMemoryAllocator
#include "tensorflow/stream_executor/host/host_platform.h"  // HostPlatform
#include "tensorflow/stream_executor/host/host_gpu_executor.h"  // HostExecutor
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/compiler/xla/client/client_library.h"  // LocalClientOptions
#include "tensorflow/compiler/xla/client/local_client.h"  
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"  // Defines PrimitiveType

using namespace std;
using namespace tensorflow;

Status PrintShape(const xla::Shape& shape);
Status PrintProgramShape(const xla::ProgramShape& program_shape);
Status CreateAndRunProgram(const string&  title, int test_case);
Status RunProgram(
  vector<xla::ExecutionInput> execution_inputs,
  const xla::ExecutableRunOptions& executable_run_options, 
  unique_ptr<xla::LocalExecutable>& local_executable);


//------------------------
int main(int argc, char* argv[]) {
  cout << "Hello World from XLA directory" << endl;

  Status r = Status::OK();
  r = CreateAndRunProgram("Executing Sequence", 0);
  if (!r.ok()) {
    cout << "Error when executing: " << r << endl;
  } else {
     cout << "Execution finished successfully" << endl;
  }

  cout << "Goodbye World from XLA directory" << endl;
  return 0;
}

//------------------------
Status CreateAndRunProgram(const string& title, int test_case) {
  cout << "***************************************************************************************" << endl;
  cout << title << ": " << test_case << endl;

  se::host::HostPlatform platform; 
  StatusOr<xla::Compiler*> sor_compiler = xla::Compiler::GetForPlatform(&platform);   
  if (!sor_compiler.ok()) {
    cout << "Error, could not get compiler: " << sor_compiler.status() << endl;
    return sor_compiler.status();
  }
  cout << "Compiler successfully retrieved for platform" << endl;
  xla::Compiler* compiler = sor_compiler.ValueOrDie();

  xla::LocalClientOptions local_client_options(
    &platform,     // Or use default: se::Platform* platform = nullptr,
    1,             // int number_of_replicas = 1,
    -1,            // int intra_op_parallelism_threads = -1,
    absl::nullopt  //const absl::optional<std::set<int>>& allowed_devices = absl::nullopt
  );

  // nullptr for default platform according to doc
  local_client_options
    .set_platform(&platform)
    // Sets the thread pool size for parallel execution of an individual operator, default is -1?
    .set_intra_op_parallelism_threads(-1)
  ;

  // Set of device IDs for which the stream executor will be created, for the given platform.
  local_client_options
    .set_allowed_devices(/*absl::optional<std::set<int>>*/{})
  ;

  // Singleton constructor-or-accessor -- returns a client for the application to issue XLA commands on.
  StatusOr<xla::LocalClient*> sor_local_client = xla::ClientLibrary::GetOrCreateLocalClient(local_client_options);
  if (!sor_local_client.ok()) {
    cout << "Could not create LocalClient: " <<  sor_local_client.status() << endl;
    return sor_local_client.status();
  }
  cout << "LocalClient created successfully" << endl;
  xla::LocalClient* local_client = sor_local_client.ValueOrDie();

  // A convenient interface for building up computations.
  // xla::XlaBuilder xlaBuilder{"__inference_return1_5<LB>_XlaMustCompile=true,config_proto=3175580994766145631,executor_type=11160318154034397263<RB>"};
  xla::XlaBuilder xla_builder{"UniqueNameHere"};
  Status s_xla_builder = xla_builder.GetCurrentStatus();
  if (!s_xla_builder.ok()) {
    cout << "Could not create XlaBuilder: " << s_xla_builder << endl;
    return s_xla_builder;
  }
  cout << "XlaBuilder created successfully" << endl;

  cout << "Now creating operations" << endl;
  xla::XlaComputation computation_add = xla::CreateScalarAddComputation(
    /*primitive_type=*/xla::S32,  // Signed int 32 bytes (int32_t)
    &xla_builder);

  // Constant scalar with value 1
  xla::Literal r1_literal = xla::LiteralUtil::CreateR0<int>(1);
  xla::LiteralSlice r1_literal_slice = xla::LiteralSlice(r1_literal);
  xla::XlaOp operand_constant_1 = ConstantLiteral(&xla_builder, r1_literal_slice);  

  // Constant 2D array
  xla::Literal r2_literal = xla::LiteralUtil::CreateR2FromArray2D((const xla::Array2D<int>){{1,2,3},{4,5,6}});
  xla::LiteralSlice r2_literal_slice = xla::LiteralSlice(r2_literal);
  xla::XlaOp operand_constant_array2d = ConstantLiteral(&xla_builder, r2_literal_slice);

  // Reduce operation of constant 2D array, adding constant scalar (1) to each dimension reduced
  xla::XlaOp operation_reduce = xla::Reduce(
    &xla_builder,
    /*operands=*/{operand_constant_array2d},
    /*init_values=*/{operand_constant_1},
    /*computation=*/computation_add,
    /*dimensions_to_reduce=*/{0}  // Results. 1 (row-based): [7, 16]. 0 (col-based): [6, 8, 10]
  );

  // Parameter of shape S32[3], using f/compiler/xla/shape_utils.h
  xla::Shape operand_param_array1d_shape = xla::ShapeUtil::MakeShape(xla::S32, {3});
  xla::XlaOp operand_param_array1d = Parameter(
    &xla_builder,
    /*int64_t parameter_number=*/0,
    /*const shape& shape=*/operand_param_array1d_shape,
    /*const std::string& name=*/"array_to_reduce");

  // Map operation
  xla::XlaOp operation_map = xla::Map(
     &xla_builder,
     {operation_reduce, operand_param_array1d},
     computation_add,
     {0}
  );

  // Print the program shape
  StatusOr<xla::ProgramShape> sor_program_shape = xla_builder.GetProgramShape();
  if (!sor_program_shape.ok()) {
    cout << "Could not retrieve program shape" << endl;
    return sor_program_shape.status();
  }
  Status program_shape_status = PrintProgramShape(sor_program_shape.ValueOrDie());
  xla::Shape program_result_layout = xla_builder.GetProgramShape().ValueOrDie().result();
  
  // Build the computation
  StatusOr<xla::XlaComputation> sor_build = xla_builder.Build();
  if (!sor_build.ok()) {
    cout << "Error during XlaBuilder::Build: " << sor_build.status() << endl;
    return sor_build.status();
  }
  cout << "XlaBuilder::Build: succesfull" << endl;
  xla::XlaComputation& xla_computation = sor_build.ValueOrDie();
  // PrintProgramShape, xla_builder.GetProgramShape() is illegal after XlaBuilder::Build 

  // Set the executable build options
  xla::ExecutableBuildOptions executable_build_options;  // tf/compiler/xla/client/executable_build_options.h
  int device_ordinal = local_client->default_device_ordinal(); 
  executable_build_options.set_device_ordinal(device_ordinal);
  executable_build_options.set_result_layout(program_result_layout);

  StatusOr<std::vector<std::unique_ptr<xla::LocalExecutable>>> sor_local_executables =
    local_client->Compile(
      xla_computation,
      /*const absl::Span<const Shape* const> argument_layouts=*/{&operand_param_array1d_shape},
      executable_build_options);
  if (!sor_local_executables.ok()) {
    cout << "Error LocalClient::Compiler failed: " << sor_local_executables.status() << endl;
    return sor_local_executables.status();
  }
  cout << "LocalClient::Compile successful" << endl;
  std::vector<std::unique_ptr<xla::LocalExecutable>>& local_executables = sor_local_executables.ValueOrDie();
  cout << "Number of executables: " << local_executables.size() << endl;
  if (local_executables.size() != 1) {
    cout << "Error, was expecting one local executables" << endl;
    return Status(tensorflow::error::NOT_FOUND, ">1 executables was unexpected");  // tf/core/protobuf/error_codes.pb.h
  }
  std::unique_ptr<xla::LocalExecutable>& local_executable = local_executables[0];
  cout << "Retrieved a single target executable, as expected" << endl;

  // Create the stream executor
  se::PluginConfig plugin_config;  // Use the defaults
  std::unique_ptr<se::internal::StreamExecutorInterface> stream_executor_impl(new se::host::HostExecutor(plugin_config));
  se::StreamExecutor stream_executor(&platform, std::move(stream_executor_impl), device_ordinal);
  se::StreamExecutorMemoryAllocator memory_allocator(&stream_executor);

  // Create run options
  xla::ExecutableRunOptions executable_run_options;
  executable_run_options
    .set_allocator(&memory_allocator)  // Argument is (se::DeviceMemoryAllocator*)
    .set_rng_seed(42)  // Hardcoding since sample, extensive algorithm here: /tf/compiler/xla/executable_run_options.h
    //.set_run_id();   // what is this
    //.set_stream();
    //.set_intra_op_thread_pool()
  ;

  // Run our program repeatedly, changing the input parameter across runs
  for (int i=10; i < 12; i++) {
    cout << "--------------" << endl << "Execution Run: " << i << endl;
    xla::ExecutionInput execution_input(operand_param_array1d_shape);  
    int param_1[] = {i, i+1, i+2};
    se::DeviceMemoryBase device_memory = se::DeviceMemory<int*>::MakeFromByteSize(param_1, sizeof(param_1));
      execution_input.SetUnownedBuffer(
        xla::ShapeIndex{},
        xla::MaybeOwningDeviceMemory(device_memory));
      vector<xla::ExecutionInput> execution_inputs;
      execution_inputs.push_back(std::move(execution_input)); 
      Status s_run_program = RunProgram(std::move(execution_inputs), executable_run_options, local_executable);
      if (!s_run_program.ok()) {
        cout << "Error on run: " << i << ", message: " << s_run_program << endl;
        return s_run_program;
      }
   }

  return Status::OK();
}

//------------------------
Status RunProgram(
  vector<xla::ExecutionInput> execution_inputs,
  const xla::ExecutableRunOptions& executable_run_options, 
  std::unique_ptr<xla::LocalExecutable>& local_executable) {

  cout << "Next statement will Run the LocalExecutable" << endl;

  StatusOr<xla::ExecutionOutput> sor_execution_output = local_executable->Run(
    std::move(execution_inputs),
    executable_run_options);
  if (!sor_execution_output.ok()) {
    cout << "LocalExecutable::Run failed: " << sor_execution_output.status() << endl;
    return sor_execution_output.status();
  }
  cout << "Run was successful: " << sor_execution_output.status() << endl;
  xla::ExecutionOutput& execution_output = sor_execution_output.ValueOrDie();
  const xla::ScopedShapedBuffer& scoped_shaped_buffer = execution_output.Result();
  cout << "Resulting ShapedBuffer.ToString: " << scoped_shaped_buffer.ToString();

  se::DeviceMemoryBase root_buffer = scoped_shaped_buffer.root_buffer();
  se::DeviceMemory<int> device_memory(root_buffer);
  cout << "DeviceMemory, ElementCount: " << device_memory.ElementCount()
    << ", IsScalar: " << device_memory.IsScalar() << endl;

  cout << "Result: ";
  int* r = static_cast<int*>(root_buffer.opaque());
  for (int i=0; i < device_memory.ElementCount(); i++) {
    cout << *(r+i);
    if (i+1 < device_memory.ElementCount()) {
      cout << ", ";
    }
  }
  cout << endl;

  return Status::OK();
}

//------------------------
Status PrintProgramShape(const xla::ProgramShape& program_shape) {
  cout << "ProgramShape.ToString: " << program_shape.ToString() << endl;
  xla::ProgramShapeProto program_shape_proto = program_shape.ToProto();
  string program_shape_proto_str;
  // https://pages.cs.wisc.edu/~starr/bots/Undermind-src/html/classgoogle_1_1protobuf_1_1io_1_1ZeroCopyOutputStream.html
  google::protobuf::io::StringOutputStream program_shape_proto_ostr(&program_shape_proto_str);
  bool success = google::protobuf::TextFormat::Print(program_shape_proto, &program_shape_proto_ostr);
  if (success) {
    cout << "ProgramShapeProto, success: " << success << ", String: [" << program_shape_proto_str << "]" << endl;
  } else {
    cout << "Error: Could not successfully Print ProgramShapeProto" << endl;
  }
  return Status::OK();
}

//------------------------
Status PrintShape(const xla::Shape& shape) {
  cout << "*** XLATesting.printShape" << endl;

  stringstream proto_sstr;
  xla::ShapeProto shape_proto = shape.ToProto();
  shape_proto.SerializeToOstream(&proto_sstr);
  string proto_str = proto_sstr.str();
  // std::stringstream sstr(std::string(stringArr,19));

  cout << "  ToString: [" << shape.ToString(true) << "]" << endl
    << "  Proto: [" << proto_str << "]" << endl;
  cout << "  element_type: "<< shape.element_type() << " (S32 is: " << xla::S32 << ")" << endl;
  cout << "  rank: " << shape.rank() << endl;
  cout << "  is_static: " << shape.is_static() << endl;
  cout << "  is_dynamic: " << shape.is_dynamic() << endl;
  cout << "  has_layout: " << shape.has_layout() << endl;
  cout << "  IsInteger: " << shape.IsInteger() << endl;
  cout << "  IsArray: " << shape.IsArray() << endl;
  cout << "  IsTuple: " << shape.IsTuple() << endl;
  cout << "  IsToken: " << shape.IsToken() << endl;
  cout << "  IsOpaque: " << shape.IsOpaque() << endl;
  cout << "  dimensions_size: " << shape.dimensions_size() << endl;
  for (int i=0; i < shape.dimensions_size(); i++) {
    cout << "    " << i << ", dimension: " << shape.dimensions(i)
      << ", is_dynamic: " << shape.is_dynamic_dimension(i) << endl;
  }
  cout << "  tuple_shapes_size: " << shape.tuple_shapes_size() << endl;
  for (int i=0; i < shape.tuple_shapes_size(); i++) {
    cout << "    " << i << ", tuple_shapes.ToString: " << shape.tuple_shapes(i).ToString() << endl;
  }
}
