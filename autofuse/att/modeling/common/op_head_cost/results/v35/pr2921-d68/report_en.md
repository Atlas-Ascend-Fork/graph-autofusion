# v35 Shared Startup Cost Calibration Results

[Case list](../../../cases.csv) | [Grouped measurements](samples.csv) | [Source and verification summary](manifest.json)

The medians of 168 groups were re-extracted from the raw profiles. The scalar conversion experiment covers
72 groups and 2,232 formal tasks; the empty-kernel and scalar-write control experiment covers 96 groups and
2,976 formal tasks. Each group has 31 formal samples. The conversion factor is **1,650.352575 cycle/us**,
with a per-round, per-core range of 1,646.794361-1,653.173507. The conversion holdout has 24 groups and a
maximum relative error of **0.6517%**. This is a measured conversion relationship, not the nominal device frequency.

| Empty-kernel observation domain | Slope (cycle/active core) | Experiment intercept (cycle) | Max relative holdout error over 24 groups |
| --- | ---: | ---: | ---: |
| Converted task time | **22.059531283** | 1314.7710 | **3.2372%** |
| AIV PMU (normalized per core) | 5.601567682 | 514.7454 | 4.6696% |
| Converted task cycles - AIV PMU | 16.457963601 | 800.0256 | 2.8349% |

The product uses only the slope from the first row. Using PMU alone misses the 16.46 cycle/active-core task-time
difference. The experiment does not attribute this difference to a specific instruction, and it does not add the
fitted intercept to the product formula. The holdout error corresponds to the experiment regression with intercept
and must not be treated as the prediction error of the production objective or a real kernel.
