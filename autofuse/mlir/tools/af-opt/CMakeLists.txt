add_executable(af-opt main.cpp)
target_link_libraries(af-opt PRIVATE
    ascendc_api_headers
    MLIROptLib
    MLIRTransforms

    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRControlFlowDialect
    MLIREmitCDialect
    MLIRFuncDialect
    MLIRIndexDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTensorDialect
    # transform dialect：设计文档 3.4 的 Schedule 迁移目标形态。这里接入的是上游
    # 方言与解释器 pass 本身（尚无自定义 transform.afir.* Op），用于在本仓构建里
    # 验证"transform dialect 在当前 LLVM 基线可用且可扩展"这一待确认项。
    MLIRTransformDialect
    MLIRTransformDialectTransforms

    MLIRAsc
    MLIRAscTransforms
    MLIRAscUtils
    MLIREmitAsc

    AFIRDialect
    AFPlanDialect
    AFIRTransforms
    AFIRToAscendC
    AfirKernelSource
    AscendTargetProfile
)

if(COMMAND llvm_update_compile_flags)
    llvm_update_compile_flags(af-opt)
endif()
if(COMMAND mlir_check_all_link_libraries)
    mlir_check_all_link_libraries(af-opt)
endif()
