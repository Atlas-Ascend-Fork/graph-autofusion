# af-afir-gen: standalone driver emitting AFIR from a built-in PlanInfo example.
# Exercises the AfirBuilder bridge core with no real backend — a template for
# adding hand-built cases.
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
