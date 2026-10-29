# Sales dataset

- train.parquet: 2024-03-28 through 2024-06-25, 90 days.
- eval.parquet: 2024-06-26 through 2024-07-02, 7 days.
- 50,000 series identified by city_id, store_id, product_id.
- Daily target: sale_amount; date: dt.

Parquet files are tracked with Git LFS. Install Git LFS before cloning, or run
`git lfs install` and `git lfs pull` after cloning to download the actual data.
