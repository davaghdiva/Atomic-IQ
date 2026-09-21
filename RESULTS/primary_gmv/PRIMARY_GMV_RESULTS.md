# Primary GMV results

The parameters in `../calibration/selected_parameters.json` are held fixed throughout January 2000–December 2024.

## Full-period GMV outcomes

The table is ordered by annualised realised volatility, the GMV objective. Every reported portfolio outcome uses the 10 bp net-return path. Expected shortfall is the positive magnitude of the mean return in the worst 5% of realised months; downside deviation uses a zero-return target. Terminal wealth is a wealth index normalised to 100 at inception and includes 10 bp transaction costs.

| method   |   realised_vol_10bp_pct |   sharpe_10bp |   max_drawdown_10bp_pct |   es95_10bp_pct |   downside_deviation_10bp_pct |   terminal_wealth_10bp_from_100 |
|:---------|------------------------:|--------------:|------------------------:|----------------:|------------------------------:|--------------------------------:|
| AIQ      |                4.123523 |      0.489247 |               16.376261 |        2.790659 |                      2.633281 |                      255.047954 |
| LS3      |                4.160439 |      0.485484 |               18.704242 |        2.851635 |                      2.658682 |                      255.105022 |
| ID_KAPPA |                4.169436 |      0.509053 |               16.605542 |        2.862378 |                      2.684996 |                      261.684797 |
| HC       |                4.171369 |      0.504789 |               16.553883 |        2.860335 |                      2.684306 |                      260.586421 |
| LS4      |                4.202174 |      0.500576 |               17.816689 |        2.936549 |                      2.737556 |                      260.359667 |
| LS5      |                4.276495 |      0.544340 |               18.269092 |        2.936470 |                      2.686028 |                      275.125847 |
| SRE_STAR |                4.330200 |      0.523165 |               18.606339 |        2.970435 |                      2.738507 |                      270.719155 |
| GS_STAR  |                4.334757 |      0.502839 |               18.162174 |        2.986201 |                      2.806292 |                      264.975774 |
| CRE      |                4.345243 |      0.519399 |               18.542654 |        2.962263 |                      2.747638 |                      270.104026 |
| NLS7     |                4.444876 |      0.482195 |               17.518407 |        3.114750 |                      2.948479 |                      262.256272 |
| NLS6     |                4.467226 |      0.486218 |               17.493624 |        3.120863 |                      2.964375 |                      264.075668 |
| NLS8     |                4.493321 |      0.491916 |               17.429294 |        3.128442 |                      2.980988 |                      266.529343 |
| LS2      |                5.601501 |      0.548943 |               16.523896 |        3.796308 |                      3.791018 |                      325.923421 |
| LS1      |                5.998545 |      0.568525 |               17.920008 |        3.939958 |                      3.995959 |                      352.268916 |
| EW       |               12.407088 |      0.426961 |               42.019727 |        8.178042 |                      8.509233 |                      485.200991 |

Annualised return and VaR are not reported. Turnover is computed from drifted pre-trade weights and is used to charge the 10 bp transaction cost, but it is not included as a standalone reported outcome.

## Panel A — GMV risk ranking

|   risk_overall_rank | method   |   vol_rank |   max_drawdown_rank |   es95_rank |   downside_deviation_rank |   risk_aggregate |
|--------------------:|:---------|-----------:|--------------------:|------------:|--------------------------:|-----------------:|
|                   1 | AIQ      |          1 |                   1 |           1 |                         1 |                4 |
|                   2 | HC       |          4 |                   3 |           3 |                         3 |               13 |
|                   3 | ID_KAPPA |          3 |                   4 |           4 |                         4 |               15 |
|                   4 | LS3      |          2 |                  14 |           2 |                         2 |               20 |
|                   5 | LS4      |          5 |                   8 |           6 |                         6 |               25 |
|                   6 | LS5      |          6 |                  11 |           5 |                         5 |               27 |
|                   7 | SRE_STAR |          7 |                  13 |           8 |                         7 |               35 |
|                   8 | GS_STAR  |          8 |                  10 |           9 |                         9 |               36 |
|                   8 | CRE      |          9 |                  12 |           7 |                         8 |               36 |
|                  10 | NLS7     |         10 |                   7 |          10 |                        10 |               37 |
|                  11 | NLS6     |         11 |                   6 |          11 |                        11 |               39 |
|                  12 | NLS8     |         12 |                   5 |          12 |                        12 |               41 |
|                  12 | LS2      |         13 |                   2 |          13 |                        13 |               41 |
|                  14 | LS1      |         14 |                   9 |          14 |                        14 |               51 |
|                  15 | EW       |         15 |                  15 |          15 |                        15 |               60 |

## Panel B — broad outcome ranking

|   broad_overall_rank | method   |   vol_rank |   max_drawdown_rank |   es95_rank |   downside_deviation_rank |   sharpe_rank |   terminal_wealth_10bp_rank |   broad_aggregate |
|---------------------:|:---------|-----------:|--------------------:|------------:|--------------------------:|--------------:|----------------------------:|------------------:|
|                    1 | AIQ      |          1 |                   1 |           1 |                         1 |            11 |                          15 |                30 |
|                    2 | ID_KAPPA |          3 |                   4 |           4 |                         4 |             6 |                          11 |                32 |
|                    2 | HC       |          4 |                   3 |           3 |                         3 |             7 |                          12 |                32 |
|                    4 | LS5      |          6 |                  11 |           5 |                         5 |             3 |                           4 |                34 |
|                    5 | SRE_STAR |          7 |                  13 |           8 |                         7 |             4 |                           5 |                44 |
|                    6 | LS2      |         13 |                   2 |          13 |                        13 |             2 |                           3 |                46 |
|                    7 | LS3      |          2 |                  14 |           2 |                         2 |            13 |                          14 |                47 |
|                    7 | LS4      |          5 |                   8 |           6 |                         6 |             9 |                          13 |                47 |
|                    7 | CRE      |          9 |                  12 |           7 |                         8 |             5 |                           6 |                47 |
|                   10 | GS_STAR  |          8 |                  10 |           9 |                         9 |             8 |                           8 |                52 |
|                   11 | LS1      |         14 |                   9 |          14 |                        14 |             1 |                           2 |                54 |
|                   12 | NLS8     |         12 |                   5 |          12 |                        12 |            10 |                           7 |                58 |
|                   13 | NLS6     |         11 |                   6 |          11 |                        11 |            12 |                           9 |                60 |
|                   14 | NLS7     |         10 |                   7 |          10 |                        10 |            14 |                          10 |                61 |
|                   15 | EW       |         15 |                  15 |          15 |                        15 |            15 |                           1 |                76 |

## Paired moving-block bootstrap: AIQ minus comparator annualised volatility

Negative values favour AIQ.

| atomic   | benchmark   |   diff_ann_vol_pp |   ci95_lo_pp |   ci95_hi_pp | ci_excludes_zero   |
|:---------|:------------|------------------:|-------------:|-------------:|:-------------------|
| AIQ      | EW          |         -8.283565 |   -10.532096 |    -6.316788 | True               |
| AIQ      | LS1         |         -1.875022 |    -2.862115 |    -1.146440 | True               |
| AIQ      | LS2         |         -1.477979 |    -2.289013 |    -0.876891 | True               |
| AIQ      | NLS8        |         -0.369799 |    -0.624351 |    -0.135662 | True               |
| AIQ      | NLS6        |         -0.343704 |    -0.584287 |    -0.120970 | True               |
| AIQ      | NLS7        |         -0.321354 |    -0.548124 |    -0.110136 | True               |
| AIQ      | CRE         |         -0.221720 |    -0.365759 |    -0.090001 | True               |
| AIQ      | GS_STAR     |         -0.211235 |    -0.420208 |    -0.007152 | True               |
| AIQ      | SRE_STAR    |         -0.206678 |    -0.340316 |    -0.089583 | True               |
| AIQ      | LS5         |         -0.152972 |    -0.270177 |    -0.039614 | True               |
| AIQ      | LS4         |         -0.078652 |    -0.179794 |     0.032323 | False              |
| AIQ      | HC          |         -0.047846 |    -0.148174 |     0.045908 | False              |
| AIQ      | ID_KAPPA    |         -0.045913 |    -0.141261 |     0.046118 | False              |
| AIQ      | LS3         |         -0.036917 |    -0.178538 |     0.110161 | False              |

The complete five-year subperiod output is in `gmv_subperiods.csv`. PSD/conditioning diagnostics are retained separately in `gmv_diagnostics.csv`.
