# ATT modeling tools

Tools are grouped by model family. [V1](v1/README_en.md) currently has no validated implementation; [v35](v35/README_en.md) contains DMA, NDDMA, MTE2 fitting, GetOpHeadCost, Gather and IsInf tools. Model designs, cases and compact results are in the [model archive](../../att/modeling/README_en.md).

The six existing tool groups moved under `v35/`; `op_head_cost` moved to `v35/common/op_head_cost`. Each project remains complete, including NDDMA predictor JSON inputs. Existing CLI arguments, targets and schemas are unchanged. Update script paths; no compatibility copies or platform dispatcher are provided. Use explicit private build/output directories and the existing shared device locks. Do not copy CANN or save tensor dumps.

Paths in commands are relative to the repository root unless stated otherwise. CMake/Python defaults resolve from their source location. Runtime architecture, API header version and model family are separate experiment properties. Templates and tiling diagnostics remain in [ATT Analyze](../att_analyze/README_en.md).
