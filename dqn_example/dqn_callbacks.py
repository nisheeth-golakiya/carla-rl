#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np

from ray.rllib.agents.callbacks import DefaultCallbacks


class DQNCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        episode.user_data["speed"] = []

    def on_episode_step(self, worker, base_env, episode, **kwargs):
        speed = worker.env.experiment.last_velocity
        episode.user_data["speed"].append(speed)

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        speeds = episode.user_data["speed"]
        if len(speeds) > 0:
            mean_speed = np.mean(episode.user_data["speed"])
        else:
            mean_speed = 0
        episode.custom_metrics["mean_speed"] = mean_speed
        episode.custom_metrics["dist_travelled"] = worker.env.experiment.distance_travelled
