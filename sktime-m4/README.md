# sktime M4 Naive baseline

The sales experiments use only `scripts/sales_naive.py`, which fits and predicts
with `sktime.forecasting.naive.NaiveForecaster(strategy="last", sp=1)`.
This is the ordinary last-value Naive baseline, not all methods in the M4 study.

Upstream source: https://github.com/mloning/sktime-m4

The upstream scripts are retained for reference. The adapter parses
`scripts/construct.py` to verify and hash the author's Naive definition without
importing the historical dependencies. Do not run the old replication scripts
as the sales experiment entry point. Install the root `requirements.txt` and run
`test_5000.py` or `test_all.py` from the main project instead.

The upstream `results/` and old build/environment files were removed because
the sales experiments do not use them. Historical copies remain in Git history.
