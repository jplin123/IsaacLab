# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import gymnasium as gym
import json
import math
import numpy as np
import os
import time
import torch
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from isaacsim.core.version import get_version

from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
from isaaclab.ui.widgets import ManagerLiveVisualizer

from .common import VecEnvStepReturn
from .manager_based_env import ManagerBasedEnv
from .manager_based_rl_env_cfg import ManagerBasedRLEnvCfg


class ManagerBasedRLEnv(ManagerBasedEnv, gym.Env):
    """The superclass for the manager-based workflow reinforcement learning-based environments.

    This class inherits from :class:`ManagerBasedEnv` and implements the core functionality for
    reinforcement learning-based environments. It is designed to be used with any RL
    library. The class is designed to be used with vectorized environments, i.e., the
    environment is expected to be run in parallel with multiple sub-environments. The
    number of sub-environments is specified using the ``num_envs``.

    Each observation from the environment is a batch of observations for each sub-
    environments. The method :meth:`step` is also expected to receive a batch of actions
    for each sub-environment.

    While the environment itself is implemented as a vectorized environment, we do not
    inherit from :class:`gym.vector.VectorEnv`. This is mainly because the class adds
    various methods (for wait and asynchronous updates) which are not required.
    Additionally, each RL library typically has its own definition for a vectorized
    environment. Thus, to reduce complexity, we directly use the :class:`gym.Env` over
    here and leave it up to library-defined wrappers to take care of wrapping this
    environment for their agents.

    Note:
        For vectorized environments, it is recommended to **only** call the :meth:`reset`
        method once before the first call to :meth:`step`, i.e. after the environment is created.
        After that, the :meth:`step` function handles the reset of terminated sub-environments.
        This is because the simulator does not support resetting individual sub-environments
        in a vectorized environment.

    """

    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: ManagerBasedRLEnvCfg
    """Configuration for the environment."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
        """
        # -- counter for curriculum
        self.common_step_counter = 0
        # -- invalid state monitor
        self._invalid_state_log_path = Path("logs/rsl_rl/sim_state_monitor_logs.jsonl")
        self._invalid_state_event_counter = 0
        # -- periodic diagnostics
        self._state_diagnostics_log_path = Path("logs/rsl_rl/state_diagnostics_logs.jsonl")
        self._state_diagnostics_event_counter = 0
        cfg_interval = getattr(cfg, "state_diagnostics_interval", 2048)
        env_interval = os.environ.get("ISAACLAB_STATE_DIAGNOSTICS_INTERVAL")
        if env_interval is not None:
            cfg_interval = int(env_interval)
        self._state_diagnostics_interval = cfg_interval if cfg_interval is not None else 0
        self._state_diagnostics_enabled = self._state_diagnostics_interval > 0
        cfg_height_thresh = getattr(cfg, "state_diagnostics_height_threshold", 1.5)
        env_height_thresh = os.environ.get("ISAACLAB_STATE_DIAGNOSTICS_HEIGHT")
        if env_height_thresh is not None:
            cfg_height_thresh = float(env_height_thresh)
        self._state_diagnostics_height_threshold = cfg_height_thresh
        # -- reward monitor placeholder (mirrors vecenv wrapper path)
        self._reward_monitor_log_path = Path("logs/rsl_rl/reward_monitor_logs.jsonl")
        # -- sanitization budget monitoring
        self._sanitization_counter = 0
        self._sanitization_window_start = 0
        self._sanitization_limit = int(os.environ.get("ISAACLAB_SANITIZATION_LIMIT", cfg.scene.num_envs))
        san_window_env = os.environ.get("ISAACLAB_SANITIZATION_WINDOW")
        self._sanitization_window = int(san_window_env) if san_window_env is not None else None
        self._sanitization_warning_emitted = False

        # initialize the episode length buffer BEFORE loading the managers to use it in mdp functions.
        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.long)

        # initialize the base class to setup the scene.
        super().__init__(cfg=cfg)
        # store the render mode
        self.render_mode = render_mode
        if self._sanitization_window is None:
            window_default = max(1, math.ceil(self.cfg.episode_length_s / self.step_dt))
            self._sanitization_window = window_default

        # initialize data and constants
        # -- set the framerate of the gym video recorder wrapper so that the playback speed of the produced video matches the simulation
        self.metadata["render_fps"] = 1 / self.step_dt

        print("[INFO]: Completed setting up the environment...")

    """
    Properties.
    """

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps."""
        return math.ceil(self.max_episode_length_s / self.step_dt)

    """
    Operations - Setup.
    """

    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # -- command manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # call the parent class to load the managers for observations and actions.
        super().load_managers()

        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def setup_manager_visualizers(self):
        """Creates live visualizers for manager terms."""

        self.manager_visualizers = {
            "action_manager": ManagerLiveVisualizer(manager=self.action_manager),
            "observation_manager": ManagerLiveVisualizer(manager=self.observation_manager),
            "command_manager": ManagerLiveVisualizer(manager=self.command_manager),
            "termination_manager": ManagerLiveVisualizer(manager=self.termination_manager),
            "reward_manager": ManagerLiveVisualizer(manager=self.reward_manager),
            "curriculum_manager": ManagerLiveVisualizer(manager=self.curriculum_manager),
        }

    """
    Operations - MDP
    """

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # sanitize articulated state before using it for rewards/terminations
        sanitized = self._sanitize_articulation_state()
        self._record_state_diagnostics(reason="sanitization" if sanitized else None, force=sanitized)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """Run rendering without stepping through the physics.

        By convention, if mode is:

        - **human**: Render to the current display and return nothing. Usually for human consumption.
        - **rgb_array**: Return a numpy.ndarray with shape (x, y, 3), representing RGB values for an
          x-by-y pixel image, suitable for turning into a video.

        Args:
            recompute: Whether to force a render even if the simulator has already rendered the scene.
                Defaults to False.

        Returns:
            The rendered image as a numpy array if mode is "rgb_array". Otherwise, returns None.

        Raises:
            RuntimeError: If mode is set to "rgb_data" and simulation render mode does not support it.
                In this case, the simulation render mode must be set to ``RenderMode.PARTIAL_RENDERING``
                or ``RenderMode.FULL_RENDERING``.
            NotImplementedError: If an unsupported rendering mode is specified.
        """
        # run a rendering step of the simulator
        # if we have rtx sensors, we do not need to render again sin
        if not self.sim.has_rtx_sensors() and not recompute:
            self.sim.render()
        # decide the rendering mode
        if self.render_mode == "human" or self.render_mode is None:
            return None
        elif self.render_mode == "rgb_array":
            # check that if any render could have happened
            if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
                raise RuntimeError(
                    f"Cannot render '{self.render_mode}' when the simulation render mode is"
                    f" '{self.sim.render_mode.name}'. Please set the simulation render mode to:"
                    f"'{self.sim.RenderMode.PARTIAL_RENDERING.name}' or '{self.sim.RenderMode.FULL_RENDERING.name}'."
                    " If running headless, make sure --enable_cameras is set."
                )
            # create the annotator if it does not exist
            if not hasattr(self, "_rgb_annotator"):
                import omni.replicator.core as rep

                # create render product
                self._render_product = rep.create.render_product(
                    self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
                )
                # create rgb annotator -- used to read data from the render product
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            # note: initially the renerer is warming up and returns empty data
            if rgb_data.size == 0:
                return np.zeros((self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3), dtype=np.uint8)
            else:
                return rgb_data[:, :, :3]
        else:
            raise NotImplementedError(
                f"Render mode '{self.render_mode}' is not supported. Please use: {self.metadata['render_modes']}."
            )

    def close(self):
        if not self._is_closed:
            # destructor is order-sensitive
            del self.command_manager
            del self.reward_manager
            del self.termination_manager
            del self.curriculum_manager
            # call the parent class to close the environment
            super().close()

    """
    Helper functions.
    """

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.single_observation_space = gym.spaces.Dict()
        for group_name, group_term_names in self.observation_manager.active_terms.items():
            # extract quantities about the group
            has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
            group_dim = self.observation_manager.group_obs_dim[group_name]
            # check if group is concatenated or not
            # if not concatenated, then we need to add each term separately as a dictionary
            if has_concatenated_obs:
                self.single_observation_space[group_name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=group_dim)
            else:
                group_term_cfgs = self.observation_manager._group_obs_term_cfgs[group_name]
                for term_name, term_dim, term_cfg in zip(group_term_names, group_dim, group_term_cfgs):
                    low = -np.inf if term_cfg.clip is None else term_cfg.clip[0]
                    high = np.inf if term_cfg.clip is None else term_cfg.clip[1]
                    self.single_observation_space[group_name] = gym.spaces.Dict(
                        {term_name: gym.spaces.Box(low=low, high=high, shape=term_dim)}
                    )
        # action space (unbounded since we don't impose any limits)
        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # update the curriculum for environments that need a reset
        self.curriculum_manager.compute(env_ids=env_ids)
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)
        # apply events such as randomizations for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- rewards manager
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- curriculum manager
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- command manager
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- termination manager
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- recorder manager
        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0

    def _sanitize_articulation_state(self) -> bool:
        """Detect NaN/Inf states in the main articulation, replace them with safe values, and log diagnostics."""
        try:
            robot = self.scene["robot"]
        except KeyError:
            return False
        num_envs = robot.data.root_pos_w.shape[0]
        device = robot.data.root_pos_w.device
        invalid_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        tensors_to_check = [
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            robot.data.root_lin_vel_w,
            robot.data.root_ang_vel_w,
            getattr(robot.data, "joint_pos", None),
            getattr(robot.data, "joint_vel", None),
            getattr(robot.data, "joint_acc", None),
            getattr(robot.data, "body_pos_w", None),
            getattr(robot.data, "body_quat_w", None),
        ]
        def _reduce_env(mask_tensor: torch.Tensor) -> torch.Tensor:
            if mask_tensor.dim() <= 1:
                return mask_tensor
            dims = tuple(range(1, mask_tensor.dim()))
            return mask_tensor.any(dim=dims)

        for tensor in tensors_to_check:
            if tensor is None:
                continue
            nan_mask = torch.isnan(tensor)
            inf_mask = torch.isinf(tensor)
            nan_mask = _reduce_env(nan_mask)
            inf_mask = _reduce_env(inf_mask)
            invalid_mask |= nan_mask
            invalid_mask |= inf_mask
        invalid_any = torch.any(invalid_mask)
        # sanitize all tracked tensors in-place
        tensors_to_sanitize = [
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            robot.data.root_lin_vel_w,
            robot.data.root_ang_vel_w,
            getattr(robot.data, "joint_pos", None),
            getattr(robot.data, "joint_vel", None),
            getattr(robot.data, "joint_acc", None),
            getattr(robot.data, "applied_torque", None),
            getattr(robot.data, "body_pos_w", None),
            getattr(robot.data, "body_quat_w", None),
        ]
        for tensor in tensors_to_sanitize:
            if tensor is None:
                continue
            torch.nan_to_num_(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        if not invalid_any:
            return False
        env_ids = torch.nonzero(invalid_mask, as_tuple=False).squeeze(-1)
        timestamp = time.time()
        self._invalid_state_event_counter += 1
        if env_ids.numel() == 0:
            return False
        env_count = env_ids.shape[0]
        sample_count = min(4, env_count)
        sample_ids_tensor = env_ids[:sample_count].detach().cpu()
        log_entry: dict[str, Any] = {
            "timestamp": timestamp,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
            "event_index": self._invalid_state_event_counter,
            "common_step": int(self.common_step_counter),
            "env_ids": env_ids.detach().cpu().tolist(),
            "sample_env_ids": sample_ids_tensor.tolist(),
            "action": "nan_sanitized",
        }
        with torch.no_grad():
            try:
                commands_tensor = self.command_manager.get_command("base_velocity").detach().cpu()
            except Exception:
                commands_tensor = None
            if commands_tensor is not None:
                log_entry["command_sample"] = commands_tensor[sample_ids_tensor].tolist()
            default_root_pos = getattr(self.cfg.scene.robot.init_state, "pos", (0.0, 0.0, 0.6))
            if default_root_pos is None:
                default_root_pos = (0.0, 0.0, 0.6)
            if not isinstance(default_root_pos, torch.Tensor):
                default_root_pos = torch.tensor(default_root_pos, dtype=robot.data.root_pos_w.dtype, device=device)
            default_root_pos = default_root_pos.unsqueeze(0).expand(env_count, -1)
            env_count = env_ids.shape[0]
            identity = torch.zeros((env_count, 4), dtype=robot.data.root_quat_w.dtype, device=device)
            identity[:, 3] = 1.0
            robot.data.root_pos_w[env_ids] = default_root_pos
            robot.data.root_quat_w[env_ids] = identity
            robot.data.root_lin_vel_w[env_ids] = 0.0
            robot.data.root_ang_vel_w[env_ids] = 0.0
            if getattr(robot.data, "joint_pos", None) is not None and hasattr(robot.data, "default_joint_pos"):
                default_joint = robot.data.default_joint_pos
                if default_joint.dim() == 1:
                    default_joint = default_joint.unsqueeze(0)
                if default_joint.shape[0] == 1:
                    default_joint = default_joint.expand(env_count, -1)
                elif default_joint.shape[0] == env_count:
                    default_joint = default_joint
                elif default_joint.shape[0] == robot.data.joint_pos.shape[0]:
                    default_joint = default_joint[env_ids]
                else:
                    default_joint = default_joint[:1].expand(env_count, -1)
                robot.data.joint_pos[env_ids] = default_joint
            if getattr(robot.data, "joint_vel", None) is not None:
                robot.data.joint_vel[env_ids] = 0.0
            if getattr(robot.data, "joint_acc", None) is not None:
                robot.data.joint_acc[env_ids] = 0.0
            if getattr(robot.data, "body_pos_w", None) is not None:
                robot.data.body_pos_w[env_ids] = 0.0
            if getattr(robot.data, "body_quat_w", None) is not None:
                body_identity = torch.zeros_like(robot.data.body_quat_w[env_ids])
                body_identity[..., 3] = 1.0
                robot.data.body_quat_w[env_ids] = body_identity
            log_entry["root_pos_sample"] = robot.data.root_pos_w[sample_ids_tensor].detach().cpu().tolist()
            log_entry["root_quat_sample"] = robot.data.root_quat_w[sample_ids_tensor].detach().cpu().tolist()
            log_entry["root_lin_vel_sample"] = robot.data.root_lin_vel_w[sample_ids_tensor].detach().cpu().tolist()
            log_entry["root_ang_vel_sample"] = robot.data.root_ang_vel_w[sample_ids_tensor].detach().cpu().tolist()
            if getattr(robot.data, "joint_pos", None) is not None:
                log_entry["joint_pos_sample"] = robot.data.joint_pos[sample_ids_tensor].detach().cpu().tolist()
            if getattr(robot.data, "joint_vel", None) is not None:
                log_entry["joint_vel_sample"] = robot.data.joint_vel[sample_ids_tensor].detach().cpu().tolist()
        self._invalid_state_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._invalid_state_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        printable_ids = ", ".join(map(str, log_entry["sample_env_ids"]))
        print(
            "[State Monitor] Sanitized invalid articulation state "
            f"(event #{self._invalid_state_event_counter}) in envs [{printable_ids}] "
            f"at step {self.common_step_counter}. Logged to {self._invalid_state_log_path}."
        )
        self._log_reward_monitor_placeholder(
            reason="sanitization", robot=robot, env_ids=env_ids, sample_ids=sample_ids_tensor
        )
        self._update_sanitization_budget(env_count)
        return True

    def _record_state_diagnostics(self, reason: str | None = None, force: bool = False):
        """Log aggregated state information periodically or when triggered."""
        try:
            robot = self.scene["robot"]
        except KeyError:
            return
        if not force:
            if not self._state_diagnostics_enabled or self._state_diagnostics_interval <= 0:
                return
            should_log = False
            reason_override = reason
            if self.common_step_counter % self._state_diagnostics_interval == 0:
                should_log = True
            elif self._state_diagnostics_height_threshold is not None:
                root_pos_w = getattr(robot.data, "root_pos_w", None)
                if root_pos_w is None:
                    return
                root_height = root_pos_w[:, 2]
                if torch.max(root_height).item() >= self._state_diagnostics_height_threshold:
                    should_log = True
                    reason_override = reason_override or "height_threshold"
            if not should_log:
                return
            reason = reason_override
        timestamp = time.time()
        self._state_diagnostics_event_counter += 1
        entry: dict[str, Any] = {
            "timestamp": timestamp,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
            "event_index": self._state_diagnostics_event_counter,
            "common_step": int(self.common_step_counter),
            "reason": reason or "interval",
        }
        with torch.no_grad():
            root_pos = robot.data.root_pos_w[:, 2]
            entry["root_height"] = {
                "min": float(torch.min(root_pos).item()),
                "max": float(torch.max(root_pos).item()),
                "mean": float(torch.mean(root_pos).item()),
            }
            torques = torch.abs(getattr(robot.data, "applied_torque", torch.empty(0)))
            if torques.numel() > 0:
                entry["joint_torque"] = {
                    "max": float(torch.max(torques).item()),
                    "mean": float(torch.mean(torques).item()),
                }
            contact_sensor = getattr(self.scene, "contact_forces", None)
            if contact_sensor is not None and hasattr(contact_sensor.data, "net_forces_w"):
                forces = torch.linalg.norm(contact_sensor.data.net_forces_w, dim=-1)
                entry["contact_force_norm"] = {
                    "max": float(torch.max(forces).item()),
                    "mean": float(torch.mean(forces).item()),
                }
        self._state_diagnostics_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_diagnostics_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        if reason:
            print(
                "[State Diagnostics] Logged event "
                f"(#{self._state_diagnostics_event_counter}, {reason}) at step {self.common_step_counter}. "
                f"Snapshot saved to {self._state_diagnostics_log_path}."
            )

    def _log_reward_monitor_placeholder(
        self, reason: str, robot, env_ids: torch.Tensor, sample_ids: torch.Tensor
    ):
        """Emit a reward-monitor style log even if the actual reward manager never ran."""
        try:
            commands = self.command_manager.get_command("base_velocity").detach().cpu()
        except Exception:
            commands = None
        snapshot = {
            "timestamp": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_index": int(self._invalid_state_event_counter),
            "common_step": int(self.common_step_counter),
            "reason": f"{reason}_pre_reward",
            "reward_sample": [],
            "reward_max": None,
            "reward_min": None,
            "reward_mean": None,
            "env_ids": env_ids.detach().cpu().tolist(),
            "sample_env_ids": sample_ids.detach().cpu().tolist(),
        }
        root_pos = robot.data.root_pos_w[sample_ids].detach().cpu().tolist()
        snapshot["root_height_sample"] = [pos[2] for pos in root_pos]
        if commands is not None:
            snapshot["command_sample"] = commands[sample_ids].tolist()
        try:
            base_vel = robot.data.root_lin_vel_b[sample_ids].detach().cpu().tolist()
            snapshot["base_lin_vel_sample"] = base_vel
        except Exception:
            pass
        self._reward_monitor_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._reward_monitor_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot) + "\n")

    def _update_sanitization_budget(self, sanitized_envs: int):
        """Abort runs where sanitization becomes chronic to save debugging time."""
        if self._sanitization_limit <= 0 or sanitized_envs <= 0:
            return
        window = max(1, self._sanitization_window)
        if self.common_step_counter - self._sanitization_window_start > window:
            self._sanitization_window_start = self.common_step_counter
            self._sanitization_counter = 0
            self._sanitization_warning_emitted = False
        self._sanitization_counter += sanitized_envs
        usage = self._sanitization_counter / max(1, self._sanitization_limit)
        if usage >= 0.5 and not self._sanitization_warning_emitted:
            print(
                "[State Monitor] Warning: sanitization budget usage "
                f"{usage * 100:.1f}% within the last {window} steps "
                f"(limit {self._sanitization_limit})."
            )
            self._sanitization_warning_emitted = True
        if self._sanitization_counter > self._sanitization_limit:
            raise RuntimeError(
                "Sanitization budget exceeded. Too many environments required NaN recovery recently. "
                "Inspect logs/rsl_rl/sim_state_monitor_logs.jsonl before resuming training."
            )
