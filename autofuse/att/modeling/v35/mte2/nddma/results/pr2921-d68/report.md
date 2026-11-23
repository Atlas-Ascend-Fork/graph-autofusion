# PR 2921 建模实测索引

[用例清单](../../cases.csv) · [逐点数据](samples.csv) · [来源与复现入口](manifest.json)

行级角色沿用原采集。无预测值表示该点未按此拟合式比较。以下误差为对应采集方法下的绝对相对误差，产品默认和全图收益需另行验收。

| 分支 | 来源 | 角色 | 点数 | 有预测的点数 | 最大误差 |
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
