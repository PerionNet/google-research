Please, read general principles in `README.MD` first.
Code for CG data is implemented in way suggested there.

<h3>How to run CG model?</h3>

1. Create `data_folder` (default is `~/tft_outputs/data/cg/`).
2. Put `df_with_all_feats_new.parquet` to `data_folder`.
3. Run
    - locally: `python -m script_download_data cg /home/<user>/tft_outputs no`
    - remotely: `run_download_data.sh` 
    
    It will create `result_input.csv`
4. Run
    - locally: `python -m script_train_fixed_params cg /home/<user>/tft_outputs no no yes`
    - remotely: `run_train_fixed_params.sh` 
    
    During first run it will perform train-valid-test split 
    and put it to create `train_campaigns.csv`, `valid_campaigns.csv` and `test_campaigns.csv`.
    If these files already exist, train-valid-test split will be taken from them.
    Finally it will create `test.csv`, `predictions.csv` files.
    
5. Run
    - locally: `python -m analyze_predictions cg /home/<user>/tft_outputs`
    - remotely: `run_analyze_predictions.sh` 
    to calculate mean error. It requires `test.csv`, `predictions.csv` and `train_campaigns.csv` and produces `result.csv`


<h3>Where to find CG code?</h3>

1. `script_download_data.preprocess_cg` - function to convert `df_with_all_feats_new.parquet`
to the format accepted by TFT.
2. `data_formatters/cg.py` is completely dedicated to CG logic.
3. Here `libs/utils.py:101` the loss is changed to skip days are not present in data
(they are marked with `target <= -1`)
4. `libs/data_utils.py` - utils to work with csv files and S3.
