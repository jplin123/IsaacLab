# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import torch
import time
import json
from pathlib import Path

from rsl_rl.env import VecEnv

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


class RslRlVecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for RSL-RL library

    To use asymmetric actor-critic, the environment instance must have the attributes :attr:`num_privileged_obs` (int).
    This is used by the learning agent to allocate buffers in the trajectory memory. Additionally, the returned
    observations should have the key "critic" which corresponds to the privileged observations. Since this is
    optional for some environments, the wrapper checks if these attributes exist. If they don't then the wrapper
    defaults to zero as number of privileged observations.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, clip_actions: float | None = None):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.
            clip_actions: The clipping value for actions. If ``None``, then no clipping is done.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        self.clip_actions = clip_actions

        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # obtain dimensions of the environment
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)
        if hasattr(self.unwrapped, "observation_manager"):
            self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
        else:
            self.num_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["policy"])
        # -- privileged observations
        if (
            hasattr(self.unwrapped, "observation_manager")
            and "critic" in self.unwrapped.observation_manager.group_obs_dim
        ):
            self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
        elif hasattr(self.unwrapped, "num_states") and "critic" in self.unwrapped.single_observation_space:
            self.num_privileged_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["critic"])
        else:
            self.num_privileged_obs = 0

        # modify the action space to the clip range
        self._modify_action_space()

        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return obs_dict["policy"], {"observations": obs_dict}

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[torch.Tensor, dict]:  # noqa: D102
        # reset the environment
        obs_dict, _ = self.env.reset()
        # return observations
        return obs_dict["policy"], {"observations": obs_dict}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # monitor rewards for numerical issues
        with torch.no_grad():
            reward_nan = torch.isnan(rew).any()
            reward_inf = torch.isinf(rew).any()
            reward_large = torch.max(torch.abs(rew)).item() > 1e3

        def _log_snapshot(reason: str):
            try:
                robot = self.unwrapped.scene["robot"]
                heights = robot.data.root_pos_w[:, 2].detach().cpu().numpy()
                base_vel = robot.data.root_lin_vel_b[:, :3].detach().cpu().numpy()
            except Exception:
                heights = None
                base_vel = None
                robot = None
            timestamp = time.time()
            event_idx = getattr(self, "_reward_monitor_event_counter", 0) + 1
            self._reward_monitor_event_counter = event_idx
            common_step = int(getattr(self.unwrapped, "common_step_counter", -1))
            try:
                episode_lengths = self.unwrapped.episode_length_buf.detach().cpu().numpy()
            except Exception:
                episode_lengths = None
            snapshot = {
                "timestamp": timestamp,
                "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
                "event_index": event_idx,
                "common_step": common_step,
                "reason": reason,
                "reward_sample": rew.detach().cpu().numpy()[:8].tolist(),
                "reward_max": float(torch.max(rew).item()),
                "reward_min": float(torch.min(rew).item()),
                "reward_mean": float(torch.mean(rew).item()),
            }
            if heights is not None:
                snapshot["root_height_sample"] = heights[:8].tolist()
            if episode_lengths is not None:
                snapshot["episode_length_sample"] = episode_lengths[:8].tolist()
            try:
                commands = self.unwrapped.command_manager.get_command("base_velocity").detach().cpu().numpy()
                snapshot["command_sample"] = commands[:8].tolist()
            except Exception:
                pass
            if base_vel is not None:
                snapshot["base_lin_vel_sample"] = base_vel[:8].tolist()
            reward_manager = getattr(self.unwrapped, "reward_manager", None)
            if reward_manager is not None:
                try:
                    term_names = list(reward_manager.active_terms)
                    step_rewards = reward_manager._step_reward.detach().clone().cpu()
                    # capture per-term magnitudes to understand explosions
                    snapshot["reward_term_max_abs"] = {
                        term: float(torch.max(torch.abs(step_rewards[:, idx])).item())
                        for idx, term in enumerate(term_names)
                    }
                    sample_count = min(4, step_rewards.shape[0])
                    term_samples: list[dict[str, float]] = []
                    for env_idx in range(sample_count):
                        term_samples.append(
                            {
                                term: float(step_rewards[env_idx, term_idx].item())
                                for term_idx, term in enumerate(term_names)
                            }
                        )
                    snapshot["reward_term_samples"] = term_samples
                except Exception:
                    pass
            log_path = getattr(
                self, "_reward_monitor_log_path", Path("logs/rsl_rl/reward_monitor_logs.jsonl")
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(snapshot) + "\n")
            printable_ts = snapshot.get("timestamp_iso", f"{timestamp:.3f}")
            reward_span = (snapshot["reward_min"], snapshot["reward_max"])
            print(
                "[Reward Monitor] Logged event "
                f"(#{event_idx}, {reason}) at step {common_step} ({printable_ts}); "
                f"reward range [{reward_span[0]:.2f}, {reward_span[1]:.2f}]. "
                f"Snapshot saved to {log_path}."
            )

        if reward_large:
            _log_snapshot("magnitude > 1e3")
        if reward_nan or reward_inf:
            reason = "NaN" if reward_nan else "Inf"
            _log_snapshot(reason)
            raise RuntimeError("Invalid reward detected (see logs above).")
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        obs = obs_dict["policy"]
        extras["observations"] = obs_dict
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # return the step information
        return obs, rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Helper functions
    """

    def _modify_action_space(self):
        """Modifies the action space to the clip range."""
        if self.clip_actions is None:
            return

        # modify the action space to the clip range
        # note: this is only possible for the box action space. we need to change it in the future for other action spaces.
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )
