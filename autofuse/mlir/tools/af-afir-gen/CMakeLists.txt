# af-afir-gen emits AFIR from a built-in PlanInfo example using AfirBuilder.
# No production backend is required; add hand-built cases to this driver as needed.
add_executable(af-afir-gen af-afir-gen.cpp)
target_link_libraries(af-afir-gen PRIVATE
  AfirBuilder
  AFIRDialect
  AFPlanDialect
  MLIRIR
  MLIRFuncDialect
  MLIRSupport
)
if(COMMAND llvm_update_compile_flags)
  llvm_update_compile_flags(af-afir-gen)
endif()
