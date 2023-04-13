from dataclasses import dataclass
from typing import cast

import gymnasium as gym
import ml.api as ml
import numpy as np
import torch
from torch import Tensor


@dataclass
class BWAction:
    """Defines the action space for the BipedalWalker task."""

    hip_1: float | Tensor
    knee_1: float | Tensor
    hip_2: float | Tensor
    knee_2: float | Tensor
    log_prob: list[float] | list[Tensor]
    value: float | Tensor | None = None
    advantage: float | Tensor | None = None
    returns: float | Tensor | None = None

    @classmethod
    def from_policy(cls, policy: Tensor, log_prob: Tensor, value: Tensor) -> "BWAction":
        assert policy.shape == (4,) and log_prob.shape == (4,) and value.shape == (1,)
        log_prob_list = log_prob.detach().cpu().tolist()
        return cls(policy[0].item(), policy[1].item(), policy[2].item(), policy[3].item(), log_prob_list, value.item())

    def to_tensor(self) -> Tensor:
        assert isinstance(self.hip_1, Tensor) and isinstance(self.knee_1, Tensor)
        assert isinstance(self.hip_2, Tensor) and isinstance(self.knee_2, Tensor)
        return torch.stack([self.hip_1, self.knee_1, self.hip_2, self.knee_2], dim=-1)


@dataclass
class BWState:
    """Defines the state space for the BipedalWalker task."""

    observation: Tensor
    reward: float | Tensor
    terminated: bool | Tensor
    truncated: bool | Tensor
    info: dict | Tensor
    reset: bool | Tensor


class BipedalWalkerEnvironment(ml.Environment[BWState, BWAction]):
    def __init__(self, hardcore: bool = False) -> None:
        super().__init__()

        self.env = gym.make("BipedalWalker-v3", hardcore=hardcore, render_mode="rgb_array")

    def _state_from_observation(
        self,
        observation: np.ndarray,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        info: dict | None = None,
        reset: bool = False,
    ) -> BWState:
        return BWState(
            observation=torch.from_numpy(observation),
            reward=reward,
            terminated=truncated,
            truncated=terminated,
            info={} if info is None else info,
            reset=reset,
        )

    def reset(self, seed: int | None = None) -> BWState:
        init_observation, init_info = self.env.reset(seed=seed)
        return self._state_from_observation(init_observation, info=init_info, reset=True)

    def render(self, state: BWState) -> np.ndarray | Tensor:
        return cast(np.ndarray, self.env.render())

    def sample_action(self) -> BWAction:
        env_sample = self.env.action_space.sample().tolist()
        return BWAction(
            hip_1=env_sample[0],
            knee_1=env_sample[1],
            hip_2=env_sample[2],
            knee_2=env_sample[3],
            log_prob=[0.0] * 4,
        )

    def step(self, action: BWAction) -> BWState:
        action_arr = np.array([action.hip_1, action.knee_1, action.hip_2, action.knee_2])
        observation_arr, reward, terminated, truncated, info = self.env.step(action_arr)
        return self._state_from_observation(observation_arr, float(reward), terminated, truncated, info)

    def terminated(self, state: BWState) -> bool:
        return cast(bool, state.terminated or state.truncated)
