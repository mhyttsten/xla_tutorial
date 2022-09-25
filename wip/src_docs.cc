

// stream_executor is not referenced/used anywhere in LocalClient::Compile
//    So it's only used for runtime part
LocalClient::Compile(); // [610 @ xla/client/local_client.cc]
  LocalService::CompileExecutables();  // [334 @ xla/service/local_service.cc]

    LocalService::GetHloModuleConfig();  // [1] [] [] [ 278 @ ./tensorflow/compiler/xla/service/local_service.cc]
      CreateExecutionOptions();  // [1] [] [] [ 322 @ ./tensorflow/compiler/xla/client/executable_build_options.cc]
      Service::CreateModuleConfig();  // [1] [] [] [ 484 @ ./tensorflow/compiler/xla/service/service.cc]

    Backend::stream_executor();  // [1] [] [] [ 381 @ ./tensorflow/compiler/xla/service/backend.cc]

   Service::BuildExecutable();  // [1040 @ xla/service/service.cc]
      HloModule::CreateFromProto();  // [578 @ ./tensorflow/compiler/xla/service/hlo_module.cc]
      HloVerifier::Run();  // [1] [] [] [ 2905 @ ./tensorflow/compiler/xla/service/hlo_verifier.cc]
      WriteStringToFile("ComputationName.14.before_optimizations.txt"); 

      compiler();  // [1] [] [] [ 260 @ ./tensorflow/compiler/xla/service/backend.h]
      CpuCompiler::RunHloPasses();  // [1042@xla/service/cpu/cpu_compiler.cc]
        CompilerTargetOptions();  // [1] [] [] [ 914 @ ./tensorflow/compiler/xla/service/cpu/cpu_compiler.cc]
        SimpleOrcJIT::InferTargetMachineForJIT();  // [1] [] [] [ 247 @ ./tensorflow/compiler/xla/service/cpu/simple_orc_jit.cc]
        CpuCompiler::RunHloPasses();  // [1] [] [] [ 885 @ ./tensorflow/compiler/xla/service/cpu/cpu_compiler.cc]
          HloModule::ToProto();  // [1] [] [] [ 492 @ ./tensorflow/compiler/xla/service/hlo_module.cc]
          LLVMTargetMachineFeatures();  // [1] [] [] [ 243 @ ./tensorflow/compiler/xla/service/cpu/target_machine_features.h]
          CpuCompiler::RunHloPassesThroughLayoutAssn();  // [1] [] [] [ 631 @ ./tensorflow/compiler/xla/service/cpu/cpu_compiler.cc]
            APPENDIX_BELOW();

          UseMlirHloLowering()/  // [383 @ ./tensorflow/compiler/xla/service/cpu/cpu_compiler.cc]
          CpuCompiler::RunHloPassesAfterLayoutAssn();  // [814 @ ./tensorflow/compiler/xla/service/cpu/cpu_compiler.cc]
            // TODO:

      compiler();  // [260@xla/service/backend.h]
      CpuCompiler::RunBackend();  // [1370@xla/service/cpu/cpu_compiler.cc]
        InitializeLLVMCommandLineOptions();  // [1121 @ ./tensorflow/compiler/xla/service/cpu/cpu_compiler.cc]
        LoadMLIRDialects();  // [1] [] [] [ 365 @ ./tensorflow/compiler/xla/service/cpu/cpu_compiler.cc]
          XLAFrameworkDialect::initialize();  // [1] [] [] [ 202 @ ./tensorflow/compiler/mlir/xla/ir/xla_framework.cc]

        SimpleOrcJIT::Create();  // [1] [] [] [ 354 @ ./tensorflow/compiler/xla/service/cpu/simple_orc_jit.cc]
          SimpleOrcJIT::InferTargetMachineForJIT();  // [1] [] [] [ 247 @ ./tensorflow/compiler/xla/service/cpu/simple_orc_jit.cc]
          CompilerFunctor();  // [1] [] [] [ 220 @ ./tensorflow/compiler/xla/service/cpu/compilder_functor.h]
          SimpleOrcJIT::SimpleOrcJIT();  // [1] [] [] [ 291 @ ./tensorflow/compiler/xla/service/cpu/simple_orc_jit.cc]

        TuplePointsToAnalysis::Run();  // [1] [] [] [ 346 @ ./tensorflow/compiler/xla/service/tuple_points_to_analysis.cc]
        HloAliasAnalysis::Run();  // [589@xla/service/hlo_alias_analysis.cc]
          HloDataflowAnalysis::Run();  // [1] [] [] [ 1842 @ ./tensorflow/compiler/xla/service/hlo_dataflow_analysis.cc]
          ComputeAliasedValues();  // [1] [] [] [ 382 @ ./tensorflow/compiler/xla/service/hlo_alias_analysis.cc]
        CallGraph::Build();  // [1] [] [] [ 503 @ ./tensorflow/compiler/xla/service/call_graph.cc]
        BufferAssigner::Run();  //  [1] [] [] [ 1242 @ ./tensorflow/compiler/xla/service/buffer_assignment.cc]
          // Lots of code here
        WriteStringtoFile("ComputationName.14.cpu_after_optimizations.txt");
        WriteStringToFile("ComputationName.14.cpu_after_optimizations-buffer-assignment.txt");  // [ 417 @ ./tensorflow/compiler/xla/service/dump.cc]

        target_machine();  // [1] [] [] [ 265 @ ./tensorflow/compiler/xla/service/cpu/simple_orc_jit.h]
        LLVMTargetMachineFeatures();  // [1] [] [] [ 243 @ ./tensorflow/compiler/xla/service/cpu/target_machine_features.h]
        CallGraph::Build();  // [1] [] [] [ 503 @ ./tensorflow/compiler/xla/service/call_graph.cc]
        IrEmitter::IrEmitter();  //  [1] [] [] [ 288 @ ./tensorflow/compiler/xla/service/cpu/ir_emitter.cc]
        IrEmitter::EmitConstantGlobals();  // [1] [] [] [ 468 @ ./tensorflow/compiler/xla/service/cpu/ir_emitter.cc]
        IrEmitter::EmitComputation();  // [1] [] [function_name_prefix: "UniqueNameHere.14"] [ 344 @ ./tensorflow/compiler/xla/service/cpu/ir_emitter.cc]

        SimpleOrcJIT::AddModule();  // [1] [] [] [ 422 @ ./tensorflow/compiler/xla/service/cpu/simple_orc_jit.cc]
        Executable();  // [ 496 @ ./tensorflow/compiler/xla/service/executable.h]
        CpuExecutable::CpuExecutable();  // [1] [] [entry_function_name: "_UniqueNameHere.14"] [ 247 @ ./tensorflow/compiler/xla/service/cpu/cpu_executable.cc]
          BufferAssignment::ToProto();  // [1] [] [] [ 1193 @ ./tensorflow/compiler/xla/service/buffer_assignment.cc]
          ModuleUniqueName();  // [1] [] [module_name: "_UniqueNameHere.14"] [ 224 @ ./tensorflow/compiler/xla/service/cpu/cpu_executable.cc]
          XlaDebugInfoManager::RegisterModule();  // [1] [] [] [ 193 @ ./tensorflow/compiler/xla/service/xla_debug_info_manager.cc]
          SimpleOrcJIT::FindCompiledSymbol();  // [1] [] [name: "_UniqueNameHere.14"] [ 440 @ ./tensorflow/compiler/xla/service/cpu/simple_orc_jit.cc]
            lambda();  // [1] [] [] [ 965 @ ./tensorflow/compiler/xla/service/cpu/cpu_compiler.cc]
              WriteStringToFile("ComputationName.14.ir-no-opt.ll");  // [1] [] [fname: "/tmp/xla_output/module_0000.UniqueNameHere.14.ir-no-opt.ll", data: "; ModuleID = '__compute_module'<NL>source_filename = "__compute_module"<NL>target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"<NL>target triple = "x86_64-unknown-darwin21.5.0"<NL><NL>@0 = private unnamed_addr constant <LB>12 x i8<RB> c"\06\00\00\00\08\00\00\00\0A\00\00\00", align 16<NL><NL>; Function Attrs: uwtable<NL>define void @UniqueNameHere.14(i8* %retval, i8* noalias %run_options, i8** noalias %params, i8** noalias %buffer_table, i8* noalias %status, i64* noalias %prof_counters) #0 {<NL>entry:<NL>  %add.0.invar_address.dim.0 = alloca i64, align 8<NL>  %0 = getelementptr inbounds i8*, i8** %buffer_table, i64 1<NL>  %1 = load i8*, i8** %0, align 8, !invariant.load !0, !dereferenceable !1, !align !2<NL>  %array_to_reduce.8 = bitcast i8* %1 to <LB>3 x i32<RB>*<NL>  %2 = getelementptr inbounds i8*, i8** %buffer_table, i64 0<NL>  %3 = load i8*, i8** %2, align 8, !invariant.load !0, !dereferenceable !1, !align !2<NL>  %add.0 = bitcast i8* %3 to <LB>3 x i32<RB>*<NL>  store i64 0, i64* %add.0.invar_address.dim.0, align 8<NL>  br label %add.0.loop_header.dim.0<NL><NL>return:                                           ; preds = %add.0.loop_exit.dim.0<NL>  ret void<NL><NL>add.0.loop_header.dim.0:                          ; preds = %add.0.loop_body.dim.0, %entry<NL>  %add.0.indvar.dim.0 = load i64, i64* %add.0.invar_address.dim.0, align 8<NL>  %4 = icmp uge i64 %add.0.indvar.dim.0, 3<NL>  br i1 %4, label %add.0.loop_exit.dim.0, label %add.0.loop_body.dim.0<NL><NL>add.0.loop_body.dim.0:                            ; preds = %add.0.loop_header.dim.0<NL>  %5 = getelementptr inbounds <LB>3 x i32<RB>, <LB>3 x i32<RB>* %array_to_reduce.8, i64 0, i64 %add.0.indvar.dim.0<NL>  %6 = load i32, i32* %5, align 4, !invariant.load !0, !noalias !3<NL>  %7 = getelementptr inbounds <LB>3 x i32<RB>, <LB>3 x i32<RB>* bitcast (<LB>12 x i8<RB>* @0 to <LB>3 x i32<RB>*), i64 0, i64 %add.0.indvar.dim.0<NL>  %8 = load i32, i32* %7, align 4, !alias.scope !7, !noalias !8<NL>  %9 = add i32 %6, %8<NL>  %10 = getelementptr inbounds <LB>3 x i32<RB>, <LB>3 x i32<RB>* %add.0, i64 0, i64 %add.0.indvar.dim.0<NL>  store i32 %9, i32* %10, align 4, !alias.scope !8, !noalias !7<NL>  %invar.inc = add nuw nsw i64 %add.0.indvar.dim.0, 1<NL>  store i64 %invar.inc, i64* %add.0.invar_address.dim.0, align 8<NL>  br label %add.0.loop_header.dim.0<NL><NL>add.0.loop_exit.dim.0:                            ; preds = %add.0.loop_header.dim.0<NL>  br label %return<NL>}<NL><NL>attributes #0 = { uwtable "denormal-fp-math"="preserve-sign" "no-frame-pointer-elim"="false" }<NL><NL>!0 = !{}<NL>!1 = !{i64 12}<NL>!2 = !{i64 16}<NL>!3 = !{!4, !6}<NL>!4 = !{!"buffer: {index:0, offset:0, size:12}", !5}<NL>!5 = !{!"XLA global AA domain"}<NL>!6 = !{!"buffer: {index:2, offset:0, size:12}", !5}<NL>!7 = !{!6}<NL>!8 = !{!4}<NL>"] [ 444 @ ./tensorflow/compiler/xla/service/dump.cc]
              WriteStringToFile("ComputationName.14.ir-no-opt-noconst.ll");  // [1] [] [fname: "/tmp/xla_output/module_0000.UniqueNameHere.14.ir-no-opt-noconst.ll", data: "; ModuleID = '__compute_module'<NL>source_filename = "__compute_module"<NL>target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"<NL>target triple = "x86_64-unknown-darwin21.5.0"<NL><NL>@0 = external dso_local unnamed_addr constant <LB>12 x i8<RB>, align 16<NL><NL>; Function Attrs: uwtable<NL>define void @UniqueNameHere.14(i8* %retval, i8* noalias %run_options, i8** noalias %params, i8** noalias %buffer_table, i8* noalias %status, i64* noalias %prof_counters) #0 {<NL>entry:<NL>  %add.0.invar_address.dim.0 = alloca i64, align 8<NL>  %0 = getelementptr inbounds i8*, i8** %buffer_table, i64 1<NL>  %1 = load i8*, i8** %0, align 8, !invariant.load !0, !dereferenceable !1, !align !2<NL>  %array_to_reduce.8 = bitcast i8* %1 to <LB>3 x i32<RB>*<NL>  %2 = getelementptr inbounds i8*, i8** %buffer_table, i64 0<NL>  %3 = load i8*, i8** %2, align 8, !invariant.load !0, !dereferenceable !1, !align !2<NL>  %add.0 = bitcast i8* %3 to <LB>3 x i32<RB>*<NL>  store i64 0, i64* %add.0.invar_address.dim.0, align 8<NL>  br label %add.0.loop_header.dim.0<NL><NL>return:                                           ; preds = %add.0.loop_exit.dim.0<NL>  ret void<NL><NL>add.0.loop_header.dim.0:                          ; preds = %add.0.loop_body.dim.0, %entry<NL>  %add.0.indvar.dim.0 = load i64, i64* %add.0.invar_address.dim.0, align 8<NL>  %4 = icmp uge i64 %add.0.indvar.dim.0, 3<NL>  br i1 %4, label %add.0.loop_exit.dim.0, label %add.0.loop_body.dim.0<NL><NL>add.0.loop_body.dim.0:                            ; preds = %add.0.loop_header.dim.0<NL>  %5 = getelementptr inbounds <LB>3 x i32<RB>, <LB>3 x i32<RB>* %array_to_reduce.8, i64 0, i64 %add.0.indvar.dim.0<NL>  %6 = load i32, i32* %5, align 4, !invariant.load !0, !noalias !3<NL>  %7 = getelementptr inbounds <LB>3 x i32<RB>, <LB>3 x i32<RB>* bitcast (<LB>12 x i8<RB>* @0 to <LB>3 x i32<RB>*), i64 0, i64 %add.0.indvar.dim.0<NL>  %8 = load i32, i32* %7, align 4, !alias.scope !7, !noalias !8<NL>  %9 = add i32 %6, %8<NL>  %10 = getelementptr inbounds <LB>3 x i32<RB>, <LB>3 x i32<RB>* %add.0, i64 0, i64 %add.0.indvar.dim.0<NL>  store i32 %9, i32* %10, align 4, !alias.scope !8, !noalias !7<NL>  %invar.inc = add nuw nsw i64 %add.0.indvar.dim.0, 1<NL>  store i64 %invar.inc, i64* %add.0.invar_address.dim.0, align 8<NL>  br label %add.0.loop_header.dim.0<NL><NL>add.0.loop_exit.dim.0:                            ; preds = %add.0.loop_header.dim.0<NL>  br label %return<NL>}<NL><NL>attributes #0 = { uwtable "denormal-fp-math"="preserve-sign" "no-frame-pointer-elim"="false" }<NL><NL>!0 = !{}<NL>!1 = !{i64 12}<NL>!2 = !{i64 16}<NL>!3 = !{!4, !6}<NL>!4 = !{!"buffer: {index:0, offset:0, size:12}", !5}<NL>!5 = !{!"XLA global AA domain"}<NL>!6 = !{!"buffer: {index:2, offset:0, size:12}", !5}<NL>!7 = !{!6}<NL>!8 = !{!4}<NL>"] [ 444 @ ./tensorflow/compiler/xla/service/dump.cc]
            CompilerFunctor::AddTargetInfoPasses();  // [1] [] [] [ 393 @ ./tensorflow/compiler/xla/service/cpu/compiler_functor.cc]
            CompilerFunctor::AddOptimizationPasses();  // [1] [] [] [ 412 @ ./tensorflow/compiler/xla/service/cpu/compiler_functor.cc]
            RewriteIRRuntimeFunctions();  // [1] [] [] [ 584 @ ./tensorflow/compiler/xla/service/cpu/llvm_ir_runtime.cc]
              RewriteCalls();  // [1] [] [fn_name: "tanhf"] [ 261 @ ./tensorflow/compiler/xla/service/cpu/llvm_ir_runtime.cc]
                // ...
              lambda();  // [1] [] [] [ 971 @ ./tensorflow/compiler/xla/service/cpu/cpu_compiler.cc]
                WriteStringToFile("UniqueNameHere.14.ir-with-opt.ll");  // [1] [] [fname: "/tmp/xla_output/module_0000.UniqueNameHere.14.ir-with-opt-noconst.ll", data: "..."]
                WriteStringToFile("UniqueNameHere.14.ir-with-opt-noconst.ll");  // [1] [] [fname: "/tmp/xla_output/module_0000.UniqueNameHere.14.ir-with-opt-noconst.ll", data: "..."]
              lambda();  // [1] [] [] [ 1092 @ ./tensorflow/compiler/xla/service/cpu/cpu_compiler.cc]
                WriteStringToFile("UniqueNameHere.14.o");  // [1] [] [fname: "/tmp/xla_output/module_0000.UniqueNameHere.14.o"]

        SimpleOrcJIT::DoneCompiling()  // [1] [] [] [ 429 @ ./tensorflow/compiler/xla/service/cpu/simple_orc_jit.cc]

      DumpingEnabledForHloModule();  // [1] [] [] [ 308 @ ./tensorflow/compiler/xla/service/dump.h]
      BufferAssignment::ToProto();  // [1] [] [] [ 1193 @ ./tensorflow/compiler/xla/service/buffer_assignment.cc]

//----------------------------------------------------------------------------
// TODO: RUNTIME PART

//----------------------------------------------------------------------------
/* APPENDIX: CpuCompiler::RunHloPassesThroughLayoutAssn()
*/

// Base class for all HLO passes
// Do not inherit it directly, use HloModulePass or HloModuleGroupPass
class HloPassInterface { // (xla/service/hlo_pass_interface.h)
  absl::string_view name();
  bool IsPassPipeline();
  void IncrementIteration();
  struct RunState { int iteration, Set<HloComputation*> changed, changed_last_iteration, changed_this_iteration};

  // Run pass on HLO module.  Returns whether it modified the module.
  StatusOr<bool> Run(HloModule* module);
  StatusOr<bool> RunOnModuleGroup(HloModuleGroup* module_group);
  Status RunOnChangedComputations(HloModule* module, RunState* run_state);
};

// Base class for passes which are module scoped
 class HloModulePass /*xla/service/hlo_pass_inference.h*/ : public HloPassInterface { 
  // Update the layout of a Shape to one that is supported by a given backend.
  void UpdateLayout(Shape* shape);
};

class HloModuleGroupPass /*xla/service/hlo_pass_inference.h*/ : public HloPassInterface {
  // Returns error, needs a group
  StatusOr<bool> Run(HloModule* module);
};

// This pass is an abstract superclass for passes that replace operations that
// match a pattern. It is intended to be subclassed, not used directly.
// This pass is useful for legalizing HLO instructions that a particular backend
// does not support into other HLO instructions.
class OpExpanderPass /*xla/service/op_exander_pass.h*/ : public HloModulePass {
  using PatternExtraFilter = std::function<bool(const HloInstruction*)>;
  OpExpanderPass(PatternExtraFilter extra_filter = nullptr);
};

Status CpuCompiler::RunHloPassesThroughLayoutAssn(
    HloModule* module,
    bool is_aot_compile,
    LLVMTargetMachineFeatures* target_machine_features,
    bool is_mlir_compile) {

  // SPMD has a separate flow, below is CPU non-spmd flow
  HloPassPipeline pipeline("HLO passes through layout assignment");

  // Add an invariant-checking pass to the pipeline. It will be run before and
  // after each HLO pass. The invariant checking pass must not mutate the graph
  // (it is required to always return "false" from its Run() method).
  pipeline.AddInvariantChecker<HloVerifier>(/*layout_sensitive=*/false,
                                            /*allow_mixed_precision=*/false);










