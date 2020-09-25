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
"""Custom formatting functions for Favorita dataset.

Defines dataset specific column definitions and data transformations.
"""
import random
from collections import namedtuple

from libs.data_utils import read_csv, write_csv

random.seed(42)

import data_formatters.base
import libs.utils as utils
import numpy as np
np.random.seed(42)
import pandas as pd
import sklearn.preprocessing

DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class FeatureName:

  DC_ID = 'data_connector_id'
  CAMPAIGN_ID = 'campaign_id'
  CAMPAIGN_EVENT = 'campaign_event'
  DATE = 'date'
  OBJECTIVE = 'objective'
  COMMON_OBJECTIVE = 'common_objective'
  NUM_ADSETS = 'num_adsets'
  BUDGET = 'budget'
  BUDGET_TYPE = 'budget_type'
  SPEND = 'spend'
  IMPRESSIONS = 'impressions'
  CLICKS = 'clicks'
  TARGET = 'target'
  TARGET_EVENT = 'target_event'
  TARGET_EVENT_OPTIMIZED_CONVERSIONS_PERCENT = 'target_event_optimized_conversions_percent'
  TARGET_EVENT_OPTIMIZED_SPEND_PERCENT = 'target_event_optimized_spend_percent'
  DAY_OF_WEEK = 'day_of_week'
  DAY_OF_MONTH = 'day_of_month'
  MONTH = 'month'
  IS_HOLIDAY = 'is_holiday'
  IS_WEEKEND = 'is_weekend'
  DOW_JONES_INDEX = 'dow_jones_index'
  CAMPAIGN_AGE = 'campaign_age'

  # All conversions
  ACTION_APP_CUSTOM_EVENT_FB_MOBILE_ADD_PAYMENT_INFO = 'action_app_custom_event_fb_mobile_add_payment_info'
  ACTION_APP_CUSTOM_EVENT_FB_MOBILE_ADD_TO_CART = 'action_app_custom_event_fb_mobile_add_to_cart'
  ACTION_APP_CUSTOM_EVENT_FB_MOBILE_ADD_TO_WISHLIST = 'action_app_custom_event_fb_mobile_add_to_wishlist'
  ACTION_APP_CUSTOM_EVENT_FB_MOBILE_COMPLETE_REGISTRATION = 'action_app_custom_event_fb_mobile_complete_registration'
  ACTION_APP_CUSTOM_EVENT_FB_MOBILE_CONTENT_VIEW = 'action_app_custom_event_fb_mobile_content_view'
  ACTION_APP_CUSTOM_EVENT_FB_MOBILE_INITIATED_CHECKOUT = 'action_app_custom_event_fb_mobile_initiated_checkout'
  ACTION_APP_CUSTOM_EVENT_FB_MOBILE_LEVEL_ACHIEVED = 'action_app_custom_event_fb_mobile_level_achieved'
  ACTION_APP_CUSTOM_EVENT_FB_MOBILE_PURCHASE = 'action_app_custom_event_fb_mobile_purchase'
  ACTION_APP_CUSTOM_EVENT_FB_MOBILE_RATE = 'action_app_custom_event_fb_mobile_rate'
  ACTION_APP_CUSTOM_EVENT_FB_MOBILE_SEARCH = 'action_app_custom_event_fb_mobile_search'
  ACTION_APP_CUSTOM_EVENT_FB_MOBILE_SPENT_CREDITS = 'action_app_custom_event_fb_mobile_spent_credits'
  ACTION_APP_CUSTOM_EVENT_FB_MOBILE_TUTORIAL_COMPLETION = 'action_app_custom_event_fb_mobile_tutorial_completion'
  ACTION_APP_CUSTOM_EVENT_OTHER = 'action_app_custom_event_other'
  ACTION_OFFSITE_CONVERSION_FB_PIXEL_ADD_PAYMENT_INFO = 'action_offsite_conversion_fb_pixel_add_payment_info'
  ACTION_OFFSITE_CONVERSION_FB_PIXEL_ADD_TO_WISHLIST = 'action_offsite_conversion_fb_pixel_add_to_wishlist'
  ACTION_OFFSITE_CONVERSION_FB_PIXEL_CONTACT = 'action_offsite_conversion_fb_pixel_contact'
  ACTION_OFFSITE_CONVERSION_FB_PIXEL_CUSTOMIZE_PRODUCT = 'action_offsite_conversion_fb_pixel_customize_product'
  ACTION_OFFSITE_CONVERSION_FB_PIXEL_DONATE = 'action_offsite_conversion_fb_pixel_donate'
  ACTION_OFFSITE_CONVERSION_FB_PIXEL_FLOW_COMPLETE = 'action_offsite_conversion_fb_pixel_flow_complete'
  ACTION_OFFSITE_CONVERSION_FB_PIXEL_LEAD = 'action_offsite_conversion_fb_pixel_lead'
  ACTION_OFFSITE_CONVERSION_FB_PIXEL_LISTING_INTERACTION = 'action_offsite_conversion_fb_pixel_listing_interaction'
  ACTION_OFFSITE_CONVERSION_FB_PIXEL_SCHEDULE = 'action_offsite_conversion_fb_pixel_schedule'
  ACTION_OFFSITE_CONVERSION_FB_PIXEL_SUBMIT_APPLICATION = 'action_offsite_conversion_fb_pixel_submit_application'
  ACTION_OMNI_ACHIEVEMENT_UNLOCKED = 'action_omni_achievement_unlocked'
  ACTION_OMNI_ADD_TO_CART = 'action_omni_add_to_cart'
  ACTION_OMNI_APP_INSTALL = 'action_omni_app_install'
  ACTION_OMNI_COMPLETE_REGISTRATION = 'action_omni_complete_registration'
  ACTION_OMNI_INITIATED_CHECKOUT = 'action_omni_initiated_checkout'
  ACTION_OMNI_LEVEL_ACHIEVED = 'action_omni_level_achieved'
  ACTION_OMNI_PURCHASE = 'action_omni_purchase'
  ACTION_OMNI_RATE = 'action_omni_rate'
  ACTION_OMNI_SEARCH = 'action_omni_search'
  ACTION_OMNI_SPEND_CREDITS = 'action_omni_spend_credits'
  ACTION_OMNI_TUTORIAL_COMPLETION = 'action_omni_tutorial_completion'
  ACTION_OMNI_VIEW_CONTENT = 'action_omni_view_content'
  CONVERSION_START_TRIAL_MOBILE_APP = "conversion_start_trial_mobile_app"
  CONVERSION_START_TRIAL_TOTAL = "conversion_start_trial_total"
  CONVERSION_SUBSCRIBE_TOTAL = "conversion_subscribe_total"

  # Windowed metrics
  CLICKS_LAST_1000_DAYS = "clicks_last_1000_days"
  CLICKS_LAST_3_DAYS = "clicks_last_3_days"
  CLICKS_LAST_5_DAYS = "clicks_last_5_days"
  CLICKS_LAST_7_DAYS = "clicks_last_7_days"
  IMPRESSIONS_LAST_1000_DAYS = "impressions_last_1000_days"
  IMPRESSIONS_LAST_3_DAYS = "impressions_last_3_days"
  IMPRESSIONS_LAST_5_DAYS = "impressions_last_5_days"
  IMPRESSIONS_LAST_7_DAYS = "impressions_last_7_days"
  SPEND_LAST_1000_DAYS = "spend_last_1000_days"
  SPEND_LAST_3_DAYS = "spend_last_3_days"
  SPEND_LAST_5_DAYS = "spend_last_5_days"
  SPEND_LAST_7_DAYS = "spend_last_7_days"
  TARGET_LAST_1000_DAYS_TO_YESTERDAY = "target_last_1000_days_to_yesterday"
  TARGET_LAST_3_DAYS_TO_YESTERDAY = "target_last_3_days_to_yesterday"
  TARGET_LAST_5_DAYS_TO_YESTERDAY = "target_last_5_days_to_yesterday"
  TARGET_LAST_7_DAYS_TO_YESTERDAY = "target_last_7_days_to_yesterday"

  # Age
  AGE_13_17 = "age_13-17"
  AGE_18_24 = "age_18-24"
  AGE_25_34 = "age_25-34"
  AGE_35_44 = "age_35-44"
  AGE_45_54 = "age_45-54"
  AGE_55_64 = "age_55-64"
  AGE_65_PLUS = "age_65+"
  AGE_ALL_AUTOMATED_APP_ADS = "age_All (Automated App Ads)"
  AGE_UNKNOWN = "age_Unknown"

  # Audience
  AUDIENCE_TYPE_GENERAL = "audience_type_general"
  AUDIENCE_TYPE_LOOKALIKE = "audience_type_lookalike"
  AUDIENCE_TYPE_OFFLINE = "audience_type_offline"
  AUDIENCE_TYPE_PIXEL = "audience_type_pixel"

  # Gender
  GENDER_ALL_AUTOMATED_APP_ADS = "gender_All (Automated App Ads)"
  GENDER_FEMALE = "gender_female"
  GENDER_MALE = "gender_male"
  GENDER_UNKNOWN = "gender_unknown"

  # Hour
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_00_00_00_00_59_59 = "hourly_stats_aggregated_by_audience_time_zone_00:00:00 - 00:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_01_00_00_01_59_59 = "hourly_stats_aggregated_by_audience_time_zone_01:00:00 - 01:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_02_00_00_02_59_59 = "hourly_stats_aggregated_by_audience_time_zone_02:00:00 - 02:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_03_00_00_03_59_59 = "hourly_stats_aggregated_by_audience_time_zone_03:00:00 - 03:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_04_00_00_04_59_59 = "hourly_stats_aggregated_by_audience_time_zone_04:00:00 - 04:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_05_00_00_05_59_59 = "hourly_stats_aggregated_by_audience_time_zone_05:00:00 - 05:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_06_00_00_06_59_59 = "hourly_stats_aggregated_by_audience_time_zone_06:00:00 - 06:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_07_00_00_07_59_59 = "hourly_stats_aggregated_by_audience_time_zone_07:00:00 - 07:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_08_00_00_08_59_59 = "hourly_stats_aggregated_by_audience_time_zone_08:00:00 - 08:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_09_00_00_09_59_59 = "hourly_stats_aggregated_by_audience_time_zone_09:00:00 - 09:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_10_00_00_10_59_59 = "hourly_stats_aggregated_by_audience_time_zone_10:00:00 - 10:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_11_00_00_11_59_59 = "hourly_stats_aggregated_by_audience_time_zone_11:00:00 - 11:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_12_00_00_12_59_59 = "hourly_stats_aggregated_by_audience_time_zone_12:00:00 - 12:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_13_00_00_13_59_59 = "hourly_stats_aggregated_by_audience_time_zone_13:00:00 - 13:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_14_00_00_14_59_59 = "hourly_stats_aggregated_by_audience_time_zone_14:00:00 - 14:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_15_00_00_15_59_59 = "hourly_stats_aggregated_by_audience_time_zone_15:00:00 - 15:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_16_00_00_16_59_59 = "hourly_stats_aggregated_by_audience_time_zone_16:00:00 - 16:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_17_00_00_17_59_59 = "hourly_stats_aggregated_by_audience_time_zone_17:00:00 - 17:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_18_00_00_18_59_59 = "hourly_stats_aggregated_by_audience_time_zone_18:00:00 - 18:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_19_00_00_19_59_59 = "hourly_stats_aggregated_by_audience_time_zone_19:00:00 - 19:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_20_00_00_20_59_59 = "hourly_stats_aggregated_by_audience_time_zone_20:00:00 - 20:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_21_00_00_21_59_59 = "hourly_stats_aggregated_by_audience_time_zone_21:00:00 - 21:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_22_00_00_22_59_59 = "hourly_stats_aggregated_by_audience_time_zone_22:00:00 - 22:59:59"
  HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_23_00_00_23_59_59 = "hourly_stats_aggregated_by_audience_time_zone_23:00:00 - 23:59:59"

  # Device
  IMPRESSION_DEVICE_ALL_AUTOMATED_APP_ADS = "impression_device_All (Automated App Ads)"
  IMPRESSION_DEVICE_ANDROID_SMARTPHONE = "impression_device_android_smartphone"
  IMPRESSION_DEVICE_ANDROID_TABLET = "impression_device_android_tablet"
  IMPRESSION_DEVICE_DESKTOP = "impression_device_desktop"
  IMPRESSION_DEVICE_IPAD = "impression_device_ipad"
  IMPRESSION_DEVICE_IPHONE = "impression_device_iphone"
  IMPRESSION_DEVICE_IPOD = "impression_device_ipod"
  IMPRESSION_DEVICE_OTHER = "impression_device_other"

  # Activity log
  CONVERSION_EVENT_UPDATED = "conversion_event_updated"
  CREATE_AD_SET_DUMMY = "create_ad_set_dummy"
  CREATE_CAMPAIGN_GROUP_DUMMY = "create_campaign_group_dummy"
  LEARNING_STATUS_LEARNING_FINISHED = "learning_status_learning finished"
  LEARNING_STATUS_LEARNING_IN_PROGRESS = "learning_status_learning_in_progress"
  UPDATE_AD_SET_BID_STRATEGY = "update_ad_set_bid_strategy"
  UPDATE_AD_SET_BUDGET_DUMMY = "update_ad_set_budget_dummy"
  UPDATE_AD_SET_DURATION = "update_ad_set_duration"
  UPDATE_AD_SET_NAME = "update_ad_set_name"
  UPDATE_AD_SET_OPTIMIZATION_GOAL = "update_ad_set_optimization_goal"
  UPDATE_AD_SET_RUN_STATUS_ACTIVE = "update_ad_set_run_status_active"
  UPDATE_AD_SET_RUN_STATUS_INACTIVE = "update_ad_set_run_status_inactive"
  UPDATE_AD_SET_TARGET_SPEC = "update_ad_set_target_spec"
  UPDATE_CAMPAIGN_BUDGET_DUMMY = "update_campaign_budget_dummy"
  UPDATE_CAMPAIGN_DELIVERY_TYPE = "update_campaign_delivery_type"
  UPDATE_CAMPAIGN_GROUP_DELIVERY_TYPE = "update_campaign_group_delivery_type"
  UPDATE_CAMPAIGN_GROUP_SPEND_CAP_DUMMY = "update_campaign_group_spend_cap_dummy"
  UPDATE_CAMPAIGN_SCHEDULE = "update_campaign_schedule"
  UPDATE_CAMPAIGN_STATUS_ACTIVE = "update_campaign_status_active"
  UPDATE_CAMPAIGN_STATUS_DELETED = "update_campaign_status_deleted"
  UPDATE_CAMPAIGN_STATUS_INACTIVE = "update_campaign_status_inactive"
  UPDATE_CAMPAIGN_STATUS_UPDATE_REQUIRED = "update_campaign_status_update_required"

  PRESENT = 'present'


Feature = namedtuple("Feature", ["name", "dtype", "tft_dtype", "tft_input_type"])


all_features = [

  Feature(FeatureName.CAMPAIGN_EVENT, str, DataTypes.CATEGORICAL, InputTypes.ID),

  Feature(FeatureName.DATE, 'date', DataTypes.DATE, InputTypes.TIME),

  Feature(FeatureName.TARGET, float, DataTypes.REAL_VALUED, InputTypes.TARGET),

  # Attributes
  Feature(FeatureName.DC_ID, int, DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
  Feature(FeatureName.CAMPAIGN_ID, int, DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
  Feature(FeatureName.TARGET_EVENT, str, DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
  Feature(FeatureName.OBJECTIVE, str, DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
  Feature(FeatureName.COMMON_OBJECTIVE, str, DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),

  # Known inputs
  Feature(FeatureName.BUDGET_TYPE, str, DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
  Feature(FeatureName.BUDGET, float, DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
  Feature(FeatureName.CAMPAIGN_AGE, int, DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
  Feature(FeatureName.DAY_OF_WEEK, int, DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
  Feature(FeatureName.DAY_OF_MONTH, int, DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
  Feature(FeatureName.MONTH, int, DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
  Feature(FeatureName.IS_HOLIDAY, int, DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
  Feature(FeatureName.IS_WEEKEND, int, DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
  Feature(FeatureName.PRESENT, int, DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),

  # General observed inputs
  Feature(FeatureName.NUM_ADSETS, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.SPEND, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.IMPRESSIONS, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.CLICKS, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.TARGET_EVENT_OPTIMIZED_SPEND_PERCENT, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.TARGET_EVENT_OPTIMIZED_CONVERSIONS_PERCENT, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

  # Conversions
  Feature(FeatureName.ACTION_APP_CUSTOM_EVENT_FB_MOBILE_ADD_PAYMENT_INFO, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_APP_CUSTOM_EVENT_FB_MOBILE_ADD_TO_CART, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_APP_CUSTOM_EVENT_FB_MOBILE_ADD_TO_WISHLIST, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_APP_CUSTOM_EVENT_FB_MOBILE_COMPLETE_REGISTRATION, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_APP_CUSTOM_EVENT_FB_MOBILE_CONTENT_VIEW, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_APP_CUSTOM_EVENT_FB_MOBILE_INITIATED_CHECKOUT, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_APP_CUSTOM_EVENT_FB_MOBILE_LEVEL_ACHIEVED, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_APP_CUSTOM_EVENT_FB_MOBILE_PURCHASE, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_APP_CUSTOM_EVENT_FB_MOBILE_RATE, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_APP_CUSTOM_EVENT_FB_MOBILE_SEARCH, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_APP_CUSTOM_EVENT_FB_MOBILE_SPENT_CREDITS, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_APP_CUSTOM_EVENT_FB_MOBILE_TUTORIAL_COMPLETION, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_APP_CUSTOM_EVENT_OTHER, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_OFFSITE_CONVERSION_FB_PIXEL_ADD_PAYMENT_INFO, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_OFFSITE_CONVERSION_FB_PIXEL_ADD_TO_WISHLIST, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_OFFSITE_CONVERSION_FB_PIXEL_CONTACT, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_OFFSITE_CONVERSION_FB_PIXEL_CUSTOMIZE_PRODUCT, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_OFFSITE_CONVERSION_FB_PIXEL_DONATE, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_OFFSITE_CONVERSION_FB_PIXEL_FLOW_COMPLETE, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_OFFSITE_CONVERSION_FB_PIXEL_LEAD, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_OFFSITE_CONVERSION_FB_PIXEL_LISTING_INTERACTION, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_OFFSITE_CONVERSION_FB_PIXEL_SCHEDULE, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_OFFSITE_CONVERSION_FB_PIXEL_SUBMIT_APPLICATION, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_OMNI_ACHIEVEMENT_UNLOCKED, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_OMNI_ADD_TO_CART, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_OMNI_APP_INSTALL, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_OMNI_COMPLETE_REGISTRATION, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_OMNI_INITIATED_CHECKOUT, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_OMNI_LEVEL_ACHIEVED, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_OMNI_PURCHASE, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_OMNI_RATE, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_OMNI_SEARCH, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_OMNI_SPEND_CREDITS, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_OMNI_TUTORIAL_COMPLETION, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.ACTION_OMNI_VIEW_CONTENT, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.CONVERSION_START_TRIAL_MOBILE_APP, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.CONVERSION_START_TRIAL_TOTAL, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.CONVERSION_SUBSCRIBE_TOTAL, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

  Feature(FeatureName.CLICKS_LAST_1000_DAYS, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.CLICKS_LAST_3_DAYS, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.CLICKS_LAST_5_DAYS, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.CLICKS_LAST_7_DAYS, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.IMPRESSIONS_LAST_1000_DAYS, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.IMPRESSIONS_LAST_3_DAYS, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.IMPRESSIONS_LAST_5_DAYS, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.IMPRESSIONS_LAST_7_DAYS, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.SPEND_LAST_1000_DAYS, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.SPEND_LAST_3_DAYS, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.SPEND_LAST_5_DAYS, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.SPEND_LAST_7_DAYS, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.TARGET_LAST_1000_DAYS_TO_YESTERDAY, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.TARGET_LAST_3_DAYS_TO_YESTERDAY, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.TARGET_LAST_5_DAYS_TO_YESTERDAY, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.TARGET_LAST_7_DAYS_TO_YESTERDAY, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

  Feature(FeatureName.AGE_13_17, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.AGE_18_24, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.AGE_25_34, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.AGE_35_44, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.AGE_45_54, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.AGE_55_64, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.AGE_65_PLUS, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.AGE_ALL_AUTOMATED_APP_ADS, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.AGE_UNKNOWN, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

  Feature(FeatureName.AUDIENCE_TYPE_GENERAL, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.AUDIENCE_TYPE_LOOKALIKE, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.AUDIENCE_TYPE_OFFLINE, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.AUDIENCE_TYPE_PIXEL, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

  Feature(FeatureName.GENDER_ALL_AUTOMATED_APP_ADS, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.GENDER_FEMALE, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.GENDER_MALE, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.GENDER_UNKNOWN, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_00_00_00_00_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_01_00_00_01_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_02_00_00_02_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_03_00_00_03_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_04_00_00_04_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_05_00_00_05_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_06_00_00_06_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_07_00_00_07_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_08_00_00_08_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_09_00_00_09_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_10_00_00_10_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_11_00_00_11_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_12_00_00_12_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_13_00_00_13_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_14_00_00_14_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_15_00_00_15_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_16_00_00_16_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_17_00_00_17_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_18_00_00_18_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_19_00_00_19_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_20_00_00_20_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_21_00_00_21_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_22_00_00_22_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.HOURLY_STATS_AGGREGATED_BY_AUDIENCE_TIME_ZONE_23_00_00_23_59_59, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

  Feature(FeatureName.IMPRESSION_DEVICE_ALL_AUTOMATED_APP_ADS, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.IMPRESSION_DEVICE_ANDROID_SMARTPHONE, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.IMPRESSION_DEVICE_ANDROID_TABLET, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.IMPRESSION_DEVICE_DESKTOP, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.IMPRESSION_DEVICE_IPAD, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.IMPRESSION_DEVICE_IPHONE, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.IMPRESSION_DEVICE_IPOD, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  Feature(FeatureName.IMPRESSION_DEVICE_OTHER, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

  # Feature(FeatureName.CONVERSION_EVENT_UPDATED, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.CREATE_AD_SET_DUMMY, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.CREATE_CAMPAIGN_GROUP_DUMMY, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.LEARNING_STATUS_LEARNING_FINISHED, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.LEARNING_STATUS_LEARNING_IN_PROGRESS, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.UPDATE_AD_SET_BID_STRATEGY, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.UPDATE_AD_SET_BUDGET_DUMMY, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.UPDATE_AD_SET_DURATION, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.UPDATE_AD_SET_NAME, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.UPDATE_AD_SET_OPTIMIZATION_GOAL, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.UPDATE_AD_SET_RUN_STATUS_ACTIVE, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.UPDATE_AD_SET_RUN_STATUS_INACTIVE, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.UPDATE_AD_SET_TARGET_SPEC, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.UPDATE_CAMPAIGN_BUDGET_DUMMY, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.UPDATE_CAMPAIGN_DELIVERY_TYPE, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # # Feature(FeatureName.UPDATE_CAMPAIGN_GROUP_DELIVERY_TYPE, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.UPDATE_CAMPAIGN_GROUP_SPEND_CAP_DUMMY, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.UPDATE_CAMPAIGN_SCHEDULE, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.UPDATE_CAMPAIGN_STATUS_ACTIVE, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.UPDATE_CAMPAIGN_STATUS_DELETED, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.UPDATE_CAMPAIGN_STATUS_INACTIVE, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
  # Feature(FeatureName.UPDATE_CAMPAIGN_STATUS_UPDATE_REQUIRED, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

  Feature(FeatureName.DOW_JONES_INDEX, float, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

]

all_features_dict = {f.name: f for f in all_features}


class CGFormatter(data_formatters.base.GenericDataFormatter):
  """Defines and formats data for the Favorita dataset.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """

  _column_definition = [(f.name, f.tft_dtype, f.tft_input_type, f.dtype) for f in all_features]

  def __init__(self):
    """Initialises formatter."""

    self.identifiers = None
    self._real_scalers = None
    self._cat_scalers = None
    self._target_scaler = None
    self._num_classes_per_cat_input = None

  def _split_series(self, s, time_steps, horizon):
    # train + valid + test
    if len(s) >= time_steps + 2 * horizon:
      return {
        'train': s.iloc[:-2 * horizon, :],
        'valid': s.iloc[-horizon - time_steps: -horizon, :],
        'test': s.iloc[-time_steps:, :],
      }
    # train + valid / test
    elif len(s) >= time_steps + horizon:
      key = 'test' if random.random() > 0.5 else 'valid'
      return {
        'train': s.iloc[: -horizon, :],
        key: s.iloc[-time_steps:, :],
      }
    # train
    elif len(s) >= time_steps:
      return {
        'train': s,
      }
    else:
      return {}

  def _split_series_1(self, s, time_steps, horizon):
    if len(s) >= time_steps + horizon:
      if random.random() > 0.5:
        return self._split_series(s, time_steps, horizon)
    key = np.random.choice(['train', 'valid', 'test'], p=[0.7, 0.15, 0.15])
    return {
      key: s,
    }

  def _train_valid_test_split(self, df, time_steps, forecast_horizon):
    df_lists = {'train': [], 'valid': [], 'test': []}
    for _, sliced in df.groupby('campaign_id'):
      sliced = sliced.sort_values('date')
      parts = self._split_series_1(sliced, time_steps, forecast_horizon)
      for k, v in parts.items():
        df_lists[k].append(v)

    for k, v in df_lists.items():
      print(f"{k} size in {len(v)}")

    dfs = {}
    for k in df_lists:
      sub_df = pd.concat(df_lists[k], axis=0).reset_index(drop=True)
      campaigns = sub_df[['campaign_id', 'date']].drop_duplicates()
      dfs[k] = campaigns

    return dfs

  def _train_valid_test_split_simple(self, df, time_steps, forecast_horizon):
    all_campaigns = df['campaign_id'].unique()
    split = np.random.choice(["train", "valid", "test"], size=len(all_campaigns), p=[0.7, 0.15, 0.15])
    return {
      k: df[df['campaign_id'].isin(all_campaigns[split == k])][['campaign_id', 'date']].drop_duplicates()
      for k in ["train", "valid", "test"]
    }

  def _load_or_perform_split(self, df, time_steps, forecast_horizon, config):
    filenames = {
      'train': 'train_campaigns.csv',
      'valid': 'valid_campaigns.csv',
      'test': 'test_campaigns.csv',
    }
    train_campaigns = read_csv(filenames['train'], config)
    if train_campaigns is None:
      print("Performing train-valid-test split...")
      campaigns = self._train_valid_test_split_simple(df, time_steps, forecast_horizon)
      for k in campaigns:
        write_csv(campaigns[k], filenames[k], config, to_csv_kwargs={'index': False})
    else:
      print("Loading existing train-valid-test split...")
      campaigns = {}
      for k, v in filenames.items():
        campaigns[k] = read_csv(v, config)
        campaigns[k]['date'] = pd.to_datetime(campaigns[k]['date'])
    for k, v in campaigns.items():
      print(f"{k} size is {len(v)} with {v['campaign_id'].nunique()} campaigns")
    return campaigns

  def split_data(self, df, config=None):
    """Splits data frame into training-validation-test data frames.

    This also calibrates scaling object, and transforms data for each split.

    Args:
      df: Source data frame to split.
      valid_boundary: Starting year for validation data
      test_boundary: Starting year for test data

    Returns:
      Tuple of transformed (train, valid, test) data.
    """

    print('Formatting train-valid-test splits.')

    fixed_params = self.get_fixed_params()
    time_steps = fixed_params['total_time_steps']
    lookback = fixed_params['num_encoder_steps']
    forecast_horizon = time_steps - lookback

    df[FeatureName.DATE] = pd.to_datetime(df[FeatureName.DATE])

    campaigns = self._load_or_perform_split(df, time_steps, forecast_horizon, config)
    train = df.merge(campaigns['train'], on=['campaign_id', 'date'])
    valid = df.merge(campaigns['valid'], on=['campaign_id', 'date'])
    test = df.merge(campaigns['test'], on=['campaign_id', 'date'])

    self.set_scalers(train, set_real=True)

    # Use all data for label encoding  to handle labels not present in training.
    self.set_scalers(df, set_real=False)

    # # Filter out identifiers not present in training (i.e. cold-started items).
    # def filter_ids(frame):
    #   identifiers = set(self.identifiers)
    #   index = frame['series_id']
    #   return frame.loc[index.apply(lambda x: x in identifiers)]
    #
    # valid = filter_ids(dfs['valid'])
    # test = filter_ids(dfs['test'])

    return (self.transform_inputs(data) for data in [train, valid, test])

  def set_scalers(self, df, set_real=True):
    """Calibrates scalers using the data supplied.

    Label encoding is applied to the entire dataset (i.e. including test),
    so that unseen labels can be handled at run-time.

    Args:
      df: Data to use to calibrate scalers.
      set_real: Whether to fit set real-valued or categorical scalers
    """
    print('Setting scalers with training data...')

    column_definitions = self.get_column_definition()
    id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                   column_definitions)
    target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                       column_definitions)

    if set_real:

      # Extract identifiers in case required
      self.identifiers = list(df[id_column].unique())

      present_df = df[df["present"] == 1]

      # Format real scalers
      self._real_scalers = {}
      for col in utils.extract_cols_from_dtype(float, self._column_definition):
        self._real_scalers[col] = (present_df[col].mean(), present_df[col].std())

      self._target_scaler = (present_df[target_column].mean(), present_df[target_column].std())

    else:
      # Format categorical scalers
      categorical_inputs = utils.extract_cols_from_data_type(
          DataTypes.CATEGORICAL, column_definitions,
          {InputTypes.ID, InputTypes.TIME})

      categorical_scalers = {}
      num_classes = []
      if self.identifiers is None:
        raise ValueError('Scale real-valued inputs first!')
      id_set = set(self.identifiers)
      valid_idx = df['campaign_id'].apply(lambda x: x in id_set)
      for col in categorical_inputs:
        # Set all to str so that we don't have mixed integer/string columns
        srs = df[col].apply(str)#.loc[valid_idx]
        categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
            srs.values)

        num_classes.append(srs.nunique())

      # Set categorical scaler outputs
      self._cat_scalers = categorical_scalers
      self._num_classes_per_cat_input = num_classes

  def transform_inputs(self, df):
    """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """
    output = df.copy()

    if self._real_scalers is None and self._cat_scalers is None:
      raise ValueError('Scalers have not been set!')

    column_definitions = self.get_column_definition()

    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    # Format real inputs
    for col in utils.extract_cols_from_dtype(float, self._column_definition):
      mean, std = self._real_scalers[col]
      output[col] = (df[col] - mean) / std

      if col == FeatureName.TARGET:
        output.loc[output[FeatureName.PRESENT] == 0, col] = -1

    # Format categorical inputs
    for col in categorical_inputs:
      string_df = df[col].apply(str)
      output[col] = self._cat_scalers[col].transform(string_df)

    return output

  def reverse_scale(self, df):
    # Format real inputs
    for col in utils.extract_cols_from_dtype(float, self._column_definition):
      mean, std = self._real_scalers[col]
      df[col] = df[col] * std + mean

  def format_predictions(self, predictions):
    """Reverts any normalisation to give predictions in original scale.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of unnormalised predictions.
    """
    output = predictions.copy()

    column_names = predictions.columns
    mean, std = self._target_scaler
    for col in column_names:
      if col not in {'forecast_time', 'identifier'}:
        output[col] = (predictions[col] * std) + mean

    return output

  # Default params
  def get_fixed_params(self):
    """Returns fixed model parameters for experiments."""

    fixed_params = {
        'total_time_steps': 14,
        'num_encoder_steps': 7,
        'num_epochs': 10,
        'early_stopping_patience': 5,
        'multiprocessing_workers': 4
    }

    return fixed_params

  def get_default_model_params(self):
    """Returns default optimised model parameters."""

    model_params = {
        'dropout_rate': 0.1,
        'hidden_layer_size': 240,
        'learning_rate': 0.001,
        'minibatch_size': 128,
        'max_gradient_norm': 100.,
        'num_heads': 4,
        'stack_size': 1
    }

    return model_params

  def get_num_samples_for_calibration(self):
    """Gets the default number of training and validation samples.

    Use to sub-sample the data for network calibration and a value of -1 uses
    all available samples.

    Returns:
      Tuple of (training samples, validation samples)
    """
    return -1, -1
