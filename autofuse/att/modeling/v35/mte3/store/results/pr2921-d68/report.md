# PR 2921 建模实测索引

[用例清单](../../cases.csv) · [逐点数据](samples.csv) · [来源与复现入口](manifest.json)

行级角色沿用原采集。无预测值表示该点未按此拟合式比较。以下误差为对应采集方法下的绝对相对误差，产品默认和全图收益需另行验收。

NORMAL 是产品系数的拟合和验收来源，NORMAL 留出最大误差为 28.79%。禁用 L2 是诊断对照，混合统计的留出最大误差超过 50%，不代表正常生产路径的拟合误差。

| 分支 | 来源 | policy | 角色 | 点数 | 有预测的点数 | 最大误差 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| StoreSmallPayload | S01 | normal | train | 6 | 6 | 22.14% |
| StoreSmallPayload | S01 | normal | holdout | 3 | 3 | 28.79% |
| StoreSmallPayload | S01 | normal | known | 4 | 4 | 36.40% |
| StoreSmallPayload | S01 | bypass | train | 6 | 6 | 54.33% |
| StoreSmallPayload | S01 | bypass | holdout | 3 | 3 | 50.92% |
| StoreSmallPayload | S01 | bypass | known | 4 | 4 | 21.64% |
