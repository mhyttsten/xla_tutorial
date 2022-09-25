load("//tensorflow:tensorflow.bzl", "filegroup")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "cc_header_only_library", "tf_cc_test")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)

filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_binary(
    name = "01_first_hlo_program",
    srcs = ["01_first_hlo_program.cc"],
    deps = [
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/core:lib_internal_impl",
        "//tensorflow/core:framework_internal_impl",
        "//tensorflow/stream_executor:stream_executor_impl",
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        "//tensorflow/compiler/xla/service/cpu:cpu_transfer_manager",
    ],
    visibility = ["//visibility:public"],
)
