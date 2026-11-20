# PR 2921 Modeling Measurement Index

[Case list](../../cases.csv) | [Point-level data](samples.csv) | [Source and reproduction entries](manifest.json)

Row-level roles follow the original collection. A dash means the point was not compared with this fitted formula.
The errors below are absolute relative errors under the corresponding collection method. Product defaults and
end-to-end gains must be accepted separately.

NORMAL is the fitting and acceptance source for the product coefficients; the NORMAL holdout maximum error is
28.79%. Disabled L2 is a diagnostic control, and the mixed-statistics holdout maximum error exceeds 50%, which does
not represent the fitting error of the normal production path.

| Branch | Source | Policy | Role | Points | Points with prediction | Max error |
| --- | --- | --- | --- | ---: | ---: | ---: |
| StoreSmallPayload | S01 | normal | train | 6 | 6 | 22.14% |
| StoreSmallPayload | S01 | normal | holdout | 3 | 3 | 28.79% |
| StoreSmallPayload | S01 | normal | known | 4 | 4 | 36.40% |
| StoreSmallPayload | S01 | bypass | train | 6 | 6 | 54.33% |
| StoreSmallPayload | S01 | bypass | holdout | 3 | 3 | 50.92% |
| StoreSmallPayload | S01 | bypass | known | 4 | 4 | 21.64% |
