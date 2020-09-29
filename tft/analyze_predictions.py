import argparse

import numpy as np
import pandas as pd

from scipy.stats import norm
from sklearn.metrics import mean_squared_log_error

from data_formatters.cg import FeatureName
from expt_settings.configs import ExperimentConfig
from libs.data_utils import read_csv


def rmsle(y_true, y_pred):
  return np.sqrt(mean_squared_log_error(y_true, y_pred))


def has_spend_changed(spend_before: np.array, spend_after: np.array, alpha: float = 0.05):
  """
  Checks whether mean of spend_after differs from mean of spend_before
  with significance level alpha.
  Assumes both spend_before and spend_after are samples produces by
  normally distributed random variables.
  Args:
    spend_before: 1-d array
    spend_after: 1-d array
    alpha: significance level of H0: mean of spend_after is equal to mean of spend_before
  Returns: True if mean of spend_after differs from mean of spend_before and False otherwise
  """
  n_before = len(spend_before)
  n_after = len(spend_after)
  mean_before_std = spend_before.std() / np.sqrt(n_before)
  mean_after_std = spend_after.std() / np.sqrt(n_after)
  diff_std = np.sqrt(mean_before_std ** 2 + mean_after_std ** 2)
  diff_dist = norm(spend_before.mean() - spend_after.mean(), diff_std)
  zero_cdf = diff_dist.cdf(0)
  return zero_cdf < alpha or zero_cdf > 1 - alpha


def main(expt_name, config):

  test_df = read_csv('test.csv', config)
  test_df['date'] = pd.to_datetime(test_df['date'])

  predictions_df = read_csv('predictions.csv', config)
  predictions_df['date'] = pd.to_datetime(predictions_df['date'])
  predictions_df['forecast_date'] = pd.to_datetime(predictions_df['forecast_date'])

  train_campaigns = read_csv("train_campaigns.csv", config)
  train_campaigns['date'] = pd.to_datetime(train_campaigns['date'])

  dfs = []

  # todo: remove after model re-train
  predictions_df = predictions_df.rename(columns={'campaign_id': FeatureName.CAMPAIGN_BG_EVENT})

  predictions_df = pd.concat([
    predictions_df,
    pd.DataFrame(
      predictions_df[FeatureName.CAMPAIGN_BG_EVENT].str.split('_', 1).tolist(),
      columns=[FeatureName.CAMPAIGN_ID, FeatureName.TARGET_EVENT],
    ),
  ], axis=1)

  for (campaign_event, campaign_id), predictions_campaign_df in predictions_df.groupby([FeatureName.CAMPAIGN_BG_EVENT, FeatureName.CAMPAIGN_ID]):
    test_campaign_df = test_df[test_df[FeatureName.CAMPAIGN_BG_EVENT] == campaign_event]
    test_campaign_df = test_campaign_df.drop(columns=[FeatureName.CAMPAIGN_ID, FeatureName.TARGET_EVENT])

    for forecast_date, sub_df in predictions_campaign_df.groupby('forecast_date'):

      sub_df[FeatureName.CAMPAIGN_ID] = sub_df[FeatureName.CAMPAIGN_ID].astype(str)

      sub_df = sub_df.set_index('date')
      sub_df = sub_df.reindex(pd.date_range(
        forecast_date - pd.Timedelta(days=14),
        sub_df.index.max(),
        freq='D',
        name='date',
      )).reset_index()

      sub_df[FeatureName.CAMPAIGN_BG_EVENT] = sub_df[FeatureName.CAMPAIGN_BG_EVENT].fillna(campaign_event)
      sub_df[FeatureName.CAMPAIGN_ID] = sub_df[FeatureName.CAMPAIGN_ID].fillna(campaign_id).astype(int)

      sub_df = sub_df.rename(columns={'target': 'target_from_pred'})

      sub_df = sub_df.merge(
        test_campaign_df,
        on=[
          FeatureName.CAMPAIGN_BG_EVENT,
          FeatureName.DATE,
        ],
      )

      sub_df = sub_df[sub_df["present"] == 1]

      sub_df = sub_df.sort_values('date')

      for col in [
        'forecast',
        FeatureName.SPEND,
        FeatureName.TARGET,
        FeatureName.BUDGET,
      ]:
        sub_df[col] = np.expm1(sub_df[col])

      sub_df.loc[sub_df['forecast'] < 0, 'forecast'] = 0
      sub_df['forecast'] = np.where(
        sub_df['forecast'].isnull(),
        sub_df[FeatureName.TARGET],
        sub_df['forecast'],
      )
      sub_df['horizon'] = sub_df['horizon'].fillna(-1).astype(int)
      sub_df['spend_real'] = sub_df[FeatureName.SPEND].cumsum()
      mean_known_spend_per_day = sub_df.loc[sub_df['horizon'] < 0, FeatureName.SPEND].mean()
      sub_df['spend_campaign_pred'] = np.where(
        sub_df['horizon'] < 0,
        sub_df[FeatureName.SPEND],
        mean_known_spend_per_day,
      )
      sub_df['spend_pred'] = sub_df['spend_campaign_pred'].cumsum()
      sub_df['conv_real'] = sub_df[FeatureName.TARGET].cumsum()
      sub_df['conv_pred'] = sub_df['forecast'].cumsum()
      sub_df['y_true'] = sub_df['spend_real'] / sub_df['conv_real']
      sub_df['y_pred'] = sub_df['spend_pred'] / sub_df['conv_pred']

      last_y_true = sub_df.loc[sub_df['horizon'] == -1, 'y_true'].iloc[-1]
      sub_df['y_pred_benchmark'] = np.where(
        sub_df['horizon'] < 0,
        sub_df['y_true'],
        last_y_true,
      )

      sub_df = sub_df.replace(np.inf, np.nan)

      if len(sub_df) < 14:
        pass

      if has_spend_changed(
          spend_before=sub_df[sub_df['horizon'] < 0][FeatureName.SPEND].values,
          spend_after=sub_df[sub_df['horizon'] >= 0][FeatureName.SPEND].values,
      ):
        pass

      if sub_df[FeatureName.TARGET].sum() < 14:
        pass

      # if campaign_id not in train_campaigns['campaign_id'].unique():
      #   pass

      dfs.append(sub_df[[
        FeatureName.CAMPAIGN_BG_EVENT, FeatureName.CAMPAIGN_ID, FeatureName.DATE,
        'horizon', FeatureName.BUDGET, FeatureName.BUDGET_TYPE,
        FeatureName.SPEND, 'spend_campaign_pred', FeatureName.TARGET, 'forecast',
        'spend_real', 'spend_pred', 'conv_real', 'conv_pred',
        'y_true', 'y_pred', 'y_pred_benchmark',
      ]])

  print(f"Sample size: {len(dfs)}")

  df = pd.concat(dfs, axis=0).reset_index(drop=True)

  errs = {}
  for horizon, sub_df in df.groupby('horizon'):
    sub_df = sub_df.dropna()
    errs[horizon] = {
      'count': len(sub_df),
      'rmsle': np.exp(rmsle(sub_df['y_true'], sub_df['y_pred'])),
      'rmsle_benchmark': np.exp(rmsle(sub_df['y_true'], sub_df['y_pred_benchmark'])),
    }
  for h, d in errs.items():
    print(h, d)

  # series_stats = pd.read_csv('series_stats.csv')
  # series_stats["was_in_valid"] = series_stats["series_days_in_data"] >= 59
  # df = df.merge(series_stats, on=['series_id'])
  # series_stats_useful_columns = ['series_days_in_data', 'series_cpd', 'campaign_days_in_data', 'campaign_cpd', 'was_in_valid']
  # df['series_cpd'] = df['series_cpd'].fillna(0)
  # df['campaign_cpd'] = df['campaign_cpd'].fillna(0)
  #
  # print(f"Number of campaigns: {len(df['campaign_id'].unique())}")
  #
  # from sklearn.linear_model import LinearRegression as LR
  # dfp = df.dropna()
  # for f in series_stats_useful_columns:
  #   print(LR().fit(dfp[[f]], dfp['err']).score(dfp[[f]], dfp['err']))

  # budget_file = '/home/roman/tft_outputs/data/campaign_budgets_processed.csv'
  # budget_df = pd.read_csv(budget_file, index_col=0)
  # budget_df['date'] = pd.to_datetime(budget_df['date'])
  # useful_budget_columns = [
  #   'campaign_start_time',
  #   'campaign_stop_time',
  #   'campaign_daily_budget',
  #   'campaign_lifetime_budget',
  #   'campaign_budget_remaining',
  #   'num_of_adsets_with_lifetime_budget',
  #   'daily_budget_of_adsets_combined',
  #   'is_cbo',
  # ]
  #
  # df = (
  #   df
  #   .merge(budget_df[['campaign_id', 'date'] + useful_budget_columns], on=['campaign_id', 'date'], how='left')
  # )

  # tmp = df.groupby(['series_id', 'campaign_id']).agg({'conversions_campaign': 'sum', 'err': 'mean'})
  # tmp = tmp.sort_values('conversions_campaign')
  #
  # tmp1 = {i: df[df['series_id'] == tmp.index[-i][0]] for i in range(1, 20)}
  # tmp2 = {i: df[df['series_id'] == tmp.index[-i][0]] for i in range(len(tmp) // 2 - 5, len(tmp) // 2 + 5)}
  # tmp3 = {i: df[df['series_id'] == tmp.index[i][0]] for i in range(20)}

  df['err'] = (df['y_true'] - df['y_pred']).abs() / df['y_true']
  print(df.groupby('horizon')['err'].mean())
  df.to_csv('results.csv', index=False)


if __name__ == "__main__":

  def get_args():
    """Gets settings from command line."""

    experiment_names = ExperimentConfig.default_experiments

    parser = argparse.ArgumentParser(description="Data download configs")
    parser.add_argument(
        "expt_name",
        metavar="e",
        type=str,
        nargs="?",
        default="volatility",
        choices=experiment_names,
        help="Experiment Name. Default={}".format(",".join(experiment_names)))
    parser.add_argument(
        "output_folder",
        metavar="f",
        type=str,
        nargs="?",
        default=".",
        help="Path to folder for data download")

    args = parser.parse_known_args()[0]

    root_folder = None if args.output_folder == "." else args.output_folder

    return args.expt_name, root_folder

  name, output_folder = get_args()

  print("Using output folder {}".format(output_folder))

  config = ExperimentConfig(name, output_folder)

  # Customise inputs to main() for new datasets.
  main(
      expt_name=name,
      config=config,
  )
