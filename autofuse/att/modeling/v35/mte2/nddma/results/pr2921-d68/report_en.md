# PR 2921 Modeling Measurement Index

[Case list](../../cases.csv) | [Point-level data](samples.csv) | [Source and reproduction entries](manifest.json)

Row-level roles follow the original collection. A dash means the point was not compared with this fitted formula.
The errors below are absolute relative errors under the corresponding collection method. Product defaults and
end-to-end gains must be accepted separately.

| Branch | Source | Role | Points | Points with prediction | Max error |
| --- | --- | --- | ---: | ---: | ---: |
| NG2 padded FP32 | S01 | train | 24 | 24 | 43.38% |
| NG2 padded FP32 | S01 | holdout | 12 | 12 | 49.46% |
| NG2 padded FP32 | S01 | known | 8 | 8 | 13.76% |
| Packed rank 3 | S02 | train | 12 | 12 | 29.20% |
| Packed rank 3 | S02 | holdout | 4 | 4 | 10.90% |
| Packed rank 3 | S02 | core_control | 9 | 9 | 43.23% |
| Sparse rank 3 | S03 | known | 6 | 6 | 16.54% |
| Sparse rank 3 | S03 | train | 6 | 6 | 7.30% |
| Sparse rank 3 | S03 | holdout | 13 | 13 | 26.10% |
| Sparse rank 4 | S04 | holdout | 16 | 16 | 48.73% |
| Sparse rank 4 | S04 | control | 2 | 2 | 311.50% |
| Small INT64 NG2 | S05 | holdout | 10 | 10 | 13.92% |
