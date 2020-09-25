# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Trains TFT based on a defined set of parameters.

Uses default parameters supplied from the configs file to train a TFT model from
scratch.

Usage:
python3 script_train_fixed_params {expt_name} {output_folder}

Command line args:
  expt_name: Name of dataset/experiment to train.
  output_folder: Root folder in which experiment is saved


"""

import argparse
import datetime as dte
import os

import data_formatters.base
import expt_settings.configs
import libs.hyperparam_opt
import libs.tft_model
import libs.utils as utils
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

from libs.data_utils import write_csv

ExperimentConfig = expt_settings.configs.ExperimentConfig
HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager
ModelClass = libs.tft_model.TemporalFusionTransformer


def format_output_column(output_df, output_col_name, test_steps):
  output_melt = output_df.melt(['forecast_time', 'identifier'], [f't+{i}' for i in range(test_steps)], 't+', output_col_name)
  output_melt['horizon'] = output_melt['t+'].str[2:].astype(int) + 1
  output_melt['date'] = output_melt['forecast_time'] + pd.to_timedelta(output_melt['horizon'], 'days')
  output_melt = output_melt.rename(columns={
    'identifier': 'campaign_id',
    'forecast_time': 'forecast_date',
  })
  return output_melt[['campaign_id', 'forecast_date', 'horizon', 'date', output_col_name]]


def format_outputs(targets, p50_forecast, test_steps):

  targets_melt = format_output_column(targets, 'target', test_steps)
  pred_melt = format_output_column(p50_forecast, 'forecast', test_steps)

  result = (
    targets_melt
    .merge(
      pred_melt,
      on=['campaign_id', 'forecast_date', 'horizon', 'date'],
      how='outer',
    )
  )

  return result


def format_test(test, data_formatter):
  data_formatter.reverse_scale(test)
  return test


def main(expt_name,
         use_gpu,
         model_folder,
         config,
         data_formatter,
         use_testing_mode=False,
         skip_train=False,
         ):
  """Trains tft based on defined model params.

  Args:
    expt_name: Name of experiment
    use_gpu: Whether to run tensorflow with GPU operations
    model_folder: Folder path where models are serialized
    data_csv_path: Path to csv file containing data
    data_formatter: Dataset-specific data fromatter (see
      expt_settings.dataformatter.GenericDataFormatter)
    use_testing_mode: Uses a smaller models and data sizes for testing purposes
      only -- switch to False to use original default settings
  """

  num_repeats = 1

  if not isinstance(data_formatter, data_formatters.base.GenericDataFormatter):
    raise ValueError(
        "Data formatters should inherit from" +
        "AbstractDataFormatter! Type={}".format(type(data_formatter)))

  # Tensorflow setup
  default_keras_session = tf.keras.backend.get_session()

  if use_gpu:
    tf_config = utils.get_default_tensorflow_config(tf_device="gpu", gpu_id=0)

  else:
    tf_config = utils.get_default_tensorflow_config(tf_device="cpu")

  print("*** Training from defined parameters for {} ***".format(expt_name))

  print("Loading & splitting data...")
  data_csv_path = config.data_csv_path
  raw_data = pd.read_csv(data_csv_path, index_col=0)
  train, valid, test = data_formatter.split_data(raw_data, config)
  train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()

  # Sets up default params
  total_steps, test_steps = config.model_steps
  fixed_params = data_formatter.get_experiment_params()
  init_params = data_formatter.get_default_model_params()
  init_params["model_folder"] = model_folder

  # Parameter overrides for testing only! Small sizes used to speed up script.
  if use_testing_mode:
    fixed_params["num_epochs"] = 1
    init_params["hidden_layer_size"] = 5
    train_samples, valid_samples = 100, 10

  # Sets up hyperparam manager
  print("*** Loading hyperparm manager ***")
  opt_manager = HyperparamOptManager({k: [init_params[k]] for k in init_params},
                                     fixed_params, model_folder)

  # Training -- one iteration only
  print("*** Running calibration ***")
  print("Params Selected:")
  for k in init_params:
    print("{}: {}".format(k, init_params[k]))

  if not skip_train:
    best_loss = np.Inf
    for _ in range(num_repeats):

      tf.reset_default_graph()
      with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

        tf.keras.backend.set_session(sess)

        params = opt_manager.get_next_parameters()
        model = ModelClass(params, use_cudnn=use_gpu)

        if not model.training_data_cached():
          model.cache_batched_data(train, "train", num_samples=train_samples)
          model.cache_batched_data(valid, "valid", num_samples=valid_samples)

        sess.run(tf.global_variables_initializer())
        model.fit()

        val_loss = model.evaluate()

        if val_loss < best_loss:
          opt_manager.update_score(params, val_loss, model)
          best_loss = val_loss

        tf.keras.backend.set_session(default_keras_session)
    best_params = opt_manager.get_best_params()
  else:
    best_params = opt_manager.get_next_parameters()

  print("*** Running tests ***")
  tf.reset_default_graph()
  with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
    tf.keras.backend.set_session(sess)
    model = ModelClass(best_params, use_cudnn=use_gpu)

    model.load(opt_manager.hyperparam_folder)

    print("Computing best validation loss")
    val_loss = model.evaluate(valid)

    print("Computing test loss")
    output_map = model.predict(test, return_targets=True)
    targets = data_formatter.format_predictions(output_map["targets"])
    p50_forecast = data_formatter.format_predictions(output_map["p50"])
    p90_forecast = data_formatter.format_predictions(output_map["p90"])

    def extract_numerical_data(data):
      """Strips out forecast time and identifier columns."""
      return data[[
          col for col in data.columns
          if col not in {"forecast_time", "identifier"}
      ]]

    p50_loss = utils.numpy_normalised_quantile_loss(
        extract_numerical_data(targets), extract_numerical_data(p50_forecast),
        0.5)
    p90_loss = utils.numpy_normalised_quantile_loss(
        extract_numerical_data(targets), extract_numerical_data(p90_forecast),
        0.9)

    tf.keras.backend.set_session(default_keras_session)

  test = format_test(test, data_formatter)
  write_csv(test, 'test.csv', config, to_csv_kwargs={'index': False})
  predictions = format_outputs(targets, p50_forecast, test_steps)
  write_csv(predictions, 'predictions.csv', config, to_csv_kwargs={'index': False})

  print("Training completed @ {}".format(dte.datetime.now()))
  print("Best validation loss = {}".format(val_loss))
  print("Params:")

  for k in best_params:
    print(k, " = ", best_params[k])
  print()
  print("Normalised Quantile Loss for Test Data: P50={}, P90={}".format(
      p50_loss.mean(), p90_loss.mean()))


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
    parser.add_argument(
        "use_gpu",
        metavar="g",
        type=str,
        nargs="?",
        choices=["yes", "no"],
        default="no",
        help="Whether to use gpu for training.")
    parser.add_argument(
      "skip_train",
      metavar="g",
      type=str,
      nargs="?",
      choices=["yes", "no"],
      default="no",
      help="Whether re-train model.")

    args = parser.parse_known_args()[0]

    root_folder = None if args.output_folder == "." else args.output_folder

    return args.expt_name, root_folder, args.use_gpu == "yes", args.skip_train == "yes"

  name, output_folder, use_tensorflow_with_gpu, skip_train = get_args()

  print("Using output folder {}".format(output_folder))

  config = ExperimentConfig(name, output_folder)
  formatter = config.make_data_formatter()

  # Customise inputs to main() for new datasets.
  main(
      expt_name=name,
      use_gpu=use_tensorflow_with_gpu,
      model_folder=os.path.join(config.model_folder, "fixed"),
      config=config,
      data_formatter=formatter,
      use_testing_mode=False,
      skip_train=skip_train)  # Change to false to use original default params
