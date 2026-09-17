/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_CORE_FEATURE_HELPER_H
#define OV_CORE_FEATURE_HELPER_H

#include <Eigen/Eigen>
#include <algorithm>
#include <memory>
#include <mutex>
#include <vector>

#include "Feature.h"
#include "FeatureDatabase.h"
#include "utils/print.h"

namespace ov_core {

/**
 * @brief Contains some nice helper functions for features.
 *
 * These functions should only depend on feature and the feature database.
 */
class FeatureHelper {

public:
  /**
   * @brief This functions will compute the disparity between common features in the two frames.
   *
   * First we find all features in the first frame.
   * Then we loop through each and find the uv of it in the next requested frame.
   * Features are skipped if no tracked feature is found (it was lost).
   * NOTE: this is on the RAW coordinates of the feature not the normalized ones.
   * NOTE: This computes the disparity over all cameras!
   *
   * @param db Feature database pointer
   * @param time0 First camera frame timestamp
   * @param time1 Second camera frame timestamp
   * @param disp_mean Average raw disparity
   * @param disp_var Variance of the disparities
   * @param total_feats Total number of common features
   */
  static void compute_disparity(std::shared_ptr<ov_core::FeatureDatabase> db, double time0, double time1, double &disp_mean,
                                double &disp_var, int &total_feats) {

    // Get features seen from the first image
    std::vector<std::shared_ptr<Feature>> feats0 = db->features_containing(time0, false, true);

    // Compute the disparity
    std::vector<double> disparities;
    for (auto &feat : feats0) {

      // Get the two uvs for both times
      for (auto &campairs : feat->timestamps) {

        // First find the two timestamps
        size_t camid = campairs.first;
        auto it0 = std::find(feat->timestamps.at(camid).begin(), feat->timestamps.at(camid).end(), time0);
        auto it1 = std::find(feat->timestamps.at(camid).begin(), feat->timestamps.at(camid).end(), time1);
        if (it0 == feat->timestamps.at(camid).end() || it1 == feat->timestamps.at(camid).end())
          continue;
        auto idx0 = std::distance(feat->timestamps.at(camid).begin(), it0);
        auto idx1 = std::distance(feat->timestamps.at(camid).begin(), it1);

        // Now lets calculate the disparity
        Eigen::Vector2f uv0 = feat->uvs.at(camid).at(idx0).block(0, 0, 2, 1);
        Eigen::Vector2f uv1 = feat->uvs.at(camid).at(idx1).block(0, 0, 2, 1);
        disparities.push_back((uv1 - uv0).norm());
      }
    }

    // If no disparities, just return
    if (disparities.size() < 2) {
      disp_mean = -1;
      disp_var = -1;
      total_feats = 0;
      return;
    }

    // Compute mean and standard deviation in respect to it
    summarize_disparities(disparities, disp_mean, disp_var);
    total_feats = (int)disparities.size();
  }

  /**
   * @brief This functions will compute the disparity over all features we have
   *
   * NOTE: this is on the RAW coordinates of the feature not the normalized ones.
   * NOTE: This computes the disparity over all cameras!
   *
   * @param db Feature database pointer
   * @param disp_mean Average raw disparity
   * @param disp_var Variance of the disparities
   * @param total_feats Total number of common features
   * @param newest_time Only compute disparity for ones older (-1 to disable)
   * @param oldest_time Only compute disparity for ones newer (-1 to disable)
   */
  static void compute_disparity(std::shared_ptr<ov_core::FeatureDatabase> db, double &disp_mean, double &disp_var, int &total_feats,
                                double newest_time = -1, double oldest_time = -1) {

    // Compute the disparity
    std::vector<double> disparities;
    for (auto &feat : db->get_internal_data()) {
      for (auto &campairs : feat.second->timestamps) {

        // Skip if only one observation
        if (campairs.second.size() < 2)
          continue;

        // Calculate the disparity using the first and last observations in
        // the requested time window. The previous implementation selected the
        // first timestamp after oldest_time and then required a later timestamp
        // to be strictly less than newest_time. That dropped a feature when
        // its newest observation landed exactly on the window boundary, which
        // is common with simulator timestamps.
        size_t camid = campairs.first;
        bool found0 = false;
        size_t num_in_window = 0;
        Eigen::Vector2f uv0 = Eigen::Vector2f::Zero();
        Eigen::Vector2f uv1 = Eigen::Vector2f::Zero();
        for (size_t idx = 0; idx < feat.second->timestamps.at(camid).size(); idx++) {
          double time = feat.second->timestamps.at(camid).at(idx);
          if ((oldest_time != -1 && time <= oldest_time) || (newest_time != -1 && time > newest_time))
            continue;
          Eigen::Vector2f uv = feat.second->uvs.at(camid).at(idx).block(0, 0, 2, 1);
          if (!found0) {
            uv0 = uv;
            found0 = true;
          }
          uv1 = uv;
          num_in_window++;
        }

        // If we found both an old and a new time, then we are good!
        if (!found0 || num_in_window < 2)
          continue;
        disparities.push_back((uv1 - uv0).norm());
      }
    }

    // If no disparities, just return
    if (disparities.size() < 2) {
      disp_mean = -1;
      disp_var = -1;
      total_feats = 0;
      return;
    }

    // Compute mean and standard deviation in respect to it
    summarize_disparities(disparities, disp_mean, disp_var);
    total_feats = (int)disparities.size();
  }

private:
  /**
   * @brief Central tendency and spread of a set of per-feature disparities.
   *
   * The static-motion gates (initialization and zero-velocity update) compare
   * this number against a few pixels. Two tracked positions of one feature are
   * occasionally mismatched on repetitive texture, and with a long window a
   * handful of such outliers adds tens of pixels to the arithmetic mean, so a
   * motionless platform reads as moving and the gate never opens. The median
   * ignores a minority of outliers while still growing with genuine motion.
   */
  static void summarize_disparities(const std::vector<double> &disparities, double &disp_mean, double &disp_var) {
    std::vector<double> sorted(disparities);
    const size_t middle = sorted.size() / 2;
    std::nth_element(sorted.begin(), sorted.begin() + middle, sorted.end());
    disp_mean = sorted[middle];
    double squared_error = 0.0;
    for (double disp_i : disparities) {
      squared_error += (disp_i - disp_mean) * (disp_i - disp_mean);
    }
    disp_var = std::sqrt(squared_error / (double)(disparities.size() - 1));
  }

  // Cannot construct this class
  FeatureHelper() {}
};

} // namespace ov_core

#endif /* OV_CORE_FEATURE_HELPER_H */
