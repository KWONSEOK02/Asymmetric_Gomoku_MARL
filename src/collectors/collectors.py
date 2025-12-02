import abc
from collections import defaultdict
import time

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, set_interaction_type, InteractionType

from src.utils.policy import _policy_t
from src.utils.log import get_log_func
from src.utils.augment import augment_transition
from src.envs.gomoku_env import GomokuEnv


# ==============================
# Reward Shaping (Potential 기반)
# ==============================

# shaping 세기 (0.0이면 shaping 비활성)
REWARD_SHAPING_LAMBDA: float = 0.01

# 상태 평가에서 공격/방어 비중
POT_A: float = 1.0   # 내 연속 수 비중
POT_B: float = 1.2   # 상대 연속 수 비중 (크면 방어 더 강조)

# True  -> 에피소드 끝에서만 shaping
# False -> 모든 스텝에서 shaping
SHAPING_TERMINAL_ONLY: bool = False


def _max_line_length_1d(line: torch.Tensor, player_value: int) -> int:
    """
    1차원 라인(가로/세로/대각선 한 줄)에서 해당 플레이어 돌의
    최장 연속 길이를 반환.
    """
    mask = (line == player_value).to(torch.int32)
    max_len = 0
    cur = 0
    for v in mask.tolist():
        if v == 1:
            cur += 1
            if cur > max_len:
                max_len = cur
        else:
            cur = 0
    return max_len


def _extract_board_from_observation(obs: torch.Tensor) -> torch.Tensor:
    """
    GomokuEnv.get_encoded_board() 구조에 맞춰 보드를 추출.

    - obs shape: (E, 3, B, B)
      채널 0: 현재 플레이어 돌 (0/1)
      채널 1: 상대 플레이어 돌 (0/1)
      채널 2: 마지막 수 위치 (0/1)

    반환:
      board: (E, B, B), 값은 {1, 0, -1}
             1  = 현재 플레이어 돌
            -1  = 상대 플레이어 돌
             0  = 빈칸
    """
    if obs.dim() != 4:
        raise ValueError(f"Expected obs shape (E, 3, B, B), got {obs.shape}")

    my = obs[:, 0]   # 현재 플레이어 돌 (0/1)
    opp = obs[:, 1]  # 상대 플레이어 돌 (0/1)
    board = my - opp  # {1, 0, -1}
    return board


def compute_potential_from_obs_batch(
    obs: torch.Tensor,
    a: float = POT_A,
    b: float = POT_B,
) -> torch.Tensor:
    """
    로컬(마지막 수 주변 4줄) 기반 potential Phi(s)를 배치 단위로 계산.

    - obs: (E, 3, B, B)
      * 채널 0: 현재 플레이어 돌 (0/1)
      * 채널 1: 상대 돌 (0/1)
      * 채널 2: 마지막 수 (0/1)

    - 반환: (E,)
      Phi(s) = a * my_local_max - b * opp_local_max
      여기서 my_local_max / opp_local_max 는
      마지막 수가 놓인 (i, j)를 지나는
      가로/세로/두 대각선 4줄에서의 최장 연속 길이.
    """
    # (E, B, B) : {1, 0, -1} 보드
    boards = _extract_board_from_observation(obs).detach().cpu()
    last_plane = obs[:, 2].detach().cpu()  # (E, B, B)

    if boards.dim() == 2:
        boards = boards.unsqueeze(0)
        last_plane = last_plane.unsqueeze(0)

    E, H, W = last_plane.shape
    phi_vals = torch.zeros(E, dtype=torch.float32)

    # ↗ 대각선 계산용 좌우 반전 보드
    flipped_boards = torch.flip(boards, dims=[2])  # (E, H, W)

    for e in range(E):
        board = boards[e]      # (H, W)
        last = last_plane[e]   # (H, W)

        # 마지막 수 좌표 찾기 (없을 수도 있음: 게임 시작 직후 등)
        coords = torch.nonzero(last, as_tuple=False)
        if coords.numel() == 0:
            # 마지막 수가 없다면 Phi(s)=0
            continue

        i, j = coords[0].tolist()

        # 가로, 세로
        row = board[i, :]    # (W,)
        col = board[:, j]    # (H,)

        # ↘ 대각선 (i, j)를 지나는 라인
        offset_main = j - i
        diag1 = board.diagonal(offset=offset_main)

        # ↗ 대각선: 좌우 반전한 보드에서의 대각선
        j_flipped = W - 1 - j
        offset_anti = j_flipped - i
        diag2 = flipped_boards[e].diagonal(offset=offset_anti)

        # 내 돌(+1), 상대 돌(-1)에 대한 최장 연속 길이 계산
        my_max = max(
            _max_line_length_1d(row, 1),
            _max_line_length_1d(col, 1),
            _max_line_length_1d(diag1, 1),
            _max_line_length_1d(diag2, 1),
        )
        opp_max = max(
            _max_line_length_1d(row, -1),
            _max_line_length_1d(col, -1),
            _max_line_length_1d(diag1, -1),
            _max_line_length_1d(diag2, -1),
        )

        phi_vals[e] = a * float(my_max) - b * float(opp_max)

    return phi_vals.to(obs.device)


# ==============================
# Transition 생성 및 라운드 로직
# ==============================

def make_transition(
    tensordict_t_minus_1: TensorDict,
    tensordict_t: TensorDict,
    tensordict_t_plus_1: TensorDict,
) -> TensorDict:
    """
    Constructs a transition tensor dictionary for a two-player game.

    상태 s_t : tensordict_t_minus_1["observation"]
    상태 s_{t+1} : tensordict_t_plus_1["observation"]

    보상:
      r = (win_t - win_{t+1}) + λ (Φ(s_{t+1}) - Φ(s_t))
    """

    # 1) 기본 승/패 보상
    reward: torch.Tensor = (
        tensordict_t.get("win").float() -
        tensordict_t_plus_1.get("win").float()
    ).unsqueeze(-1)  # shape: (E, 1)

    # 2) potential-based shaping
    obs_prev = tensordict_t_minus_1.get("observation", None)
    obs_next = tensordict_t_plus_1.get("observation", None)
    done_flag = tensordict_t_plus_1.get("done", None)

    if (
        REWARD_SHAPING_LAMBDA != 0.0
        and obs_prev is not None
        and obs_next is not None
    ):
        if SHAPING_TERMINAL_ONLY and done_flag is not None:
            # --- 에피소드 종료 시점에서만 shaping ---
            done_mask = done_flag.view(-1).bool()
            if done_mask.any():
                phi_prev = compute_potential_from_obs_batch(
                    obs_prev[done_mask]
                )  # (E_done,)
                phi_next = compute_potential_from_obs_batch(
                    obs_next[done_mask]
                )  # (E_done,)
                shaping = REWARD_SHAPING_LAMBDA * (phi_next - phi_prev)  # (E_done,)

                reward_flat = reward.view(-1, 1)
                reward_flat[done_mask, 0] = reward_flat[done_mask, 0] + shaping
                reward = reward_flat.view_as(reward)
        else:
            # --- 모든 스텝에 shaping 적용 ---
            phi_prev = compute_potential_from_obs_batch(obs_prev)   # (E,)
            phi_next = compute_potential_from_obs_batch(obs_next)   # (E,)
            shaping = REWARD_SHAPING_LAMBDA * (phi_next - phi_prev)  # (E,)
            reward = reward + shaping.unsqueeze(-1)

    # 3) transition tensordict 구성
    transition: TensorDict = tensordict_t_minus_1.select(
        "observation",
        "action_mask",
        "action",
        "sample_log_prob",
        "state_value",
        strict=False,
    )
    transition.set(
        "next",
        tensordict_t_plus_1.select(
            "observation", "action_mask", "state_value", strict=False
        ),
    )
    transition.set(("next", "reward"), reward)
    done = tensordict_t_plus_1["done"] | tensordict_t["done"]
    transition.set(("next", "done"), done)
    return transition


def round(
    env: GomokuEnv,
    policy_black: _policy_t,
    policy_white: _policy_t,
    tensordict_t_minus_1: TensorDict,
    tensordict_t: TensorDict,
    return_black_transitions: bool = True,
    return_white_transitions: bool = True,
):
    """Executes two sequential steps in the Gomoku environment."""

    # 첫 수: white가 둔다 (t -> t+1)
    tensordict_t_plus_1 = env.step_and_maybe_reset(tensordict=tensordict_t)
    with set_interaction_type(type=InteractionType.RANDOM):
        tensordict_t_plus_1 = policy_white(tensordict_t_plus_1)

    if return_white_transitions:
        transition_white = make_transition(
            tensordict_t_minus_1, tensordict_t, tensordict_t_plus_1
        )
        invalid: torch.Tensor = tensordict_t_minus_1["done"]
        transition_white["next", "done"] = (
            invalid | transition_white["next", "done"]
        )
        transition_white.set("invalid", invalid)
    else:
        transition_white = None

    # 둘째 수: black이 둔다 (t+1 -> t+2)
    tensordict_t_plus_2 = env.step_and_maybe_reset(
        tensordict_t_plus_1,
        env_mask=~tensordict_t_plus_1.get("done"),
    )
    with set_interaction_type(type=InteractionType.RANDOM):
        tensordict_t_plus_2 = policy_black(tensordict_t_plus_2)

    if return_black_transitions:
        transition_black = make_transition(
            tensordict_t, tensordict_t_plus_1, tensordict_t_plus_2
        )
        transition_black.set(
            "invalid",
            torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
        )
    else:
        transition_black = None

    return (
        transition_black,
        transition_white,
        tensordict_t_plus_1,
        tensordict_t_plus_2,
    )


def self_play_step(
    env: GomokuEnv,
    policy: _policy_t,
    tensordict_t_minus_1: TensorDict,
    tensordict_t: TensorDict,
):
    """Executes a single step of self-play."""

    tensordict_t_plus_1 = env.step_and_maybe_reset(
        tensordict=tensordict_t
    )
    with set_interaction_type(type=InteractionType.RANDOM):
        tensordict_t_plus_1 = policy(tensordict_t_plus_1)

    transition = make_transition(
        tensordict_t_minus_1, tensordict_t, tensordict_t_plus_1
    )
    return (
        transition,
        tensordict_t,
        tensordict_t_plus_1,
    )


# ==============================
# Collector 추상 클래스 및 구현
# ==============================

class Collector(abc.ABC):
    @abc.abstractmethod
    def rollout(self, steps: int):
        """Return transitions and info dict."""
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class SelfPlayCollector(Collector):
    def __init__(
        self,
        env: GomokuEnv,
        policy: _policy_t,
        out_device=None,
        augment: bool = False,
    ):
        """Initializes a collector for self-play data."""
        self._env = env
        self._policy = policy
        self._out_device = out_device or self._env.device
        self._augment = augment
        self._t = None
        self._t_minus_1 = None

    def reset(self):
        self._env.reset()
        self._t = None
        self._t_minus_1 = None

    @torch.no_grad()
    def rollout(self, steps: int) -> tuple[TensorDict, dict]:
        """Executes a rollout in the environment (self-play)."""

        info: defaultdict[str, float] = defaultdict(float)
        self._env.set_post_step(get_log_func(info))
        tensordicts = []
        start = time.perf_counter()

        if self._t_minus_1 is None and self._t is None:
            self._t_minus_1 = self._env.reset()
            with set_interaction_type(type=InteractionType.RANDOM):
                self._t_minus_1 = self._policy(self._t_minus_1)
            self._t = self._env.step(self._t_minus_1)
            with set_interaction_type(type=InteractionType.RANDOM):
                self._t = self._policy(self._t)

        for i in range(steps - 1):
            (
                transition,
                self._t_minus_1,
                self._t,
            ) = self_play_step(self._env, self._policy, self._t_minus_1, self._t)

            if i == steps - 2:
                transition["next", "done"] = torch.ones(
                    transition["next", "done"].shape,
                    dtype=torch.bool,
                    device=transition.device,
                )

            if self._augment:
                transition = augment_transition(transition)

            tensordicts.append(transition.to(self._out_device))

        end = time.perf_counter()
        fps = (steps * self._env.num_envs) / (end - start)
        self._env.set_post_step(None)

        tensordicts = torch.stack(tensordicts, dim=-1)
        info.update({"fps": fps})
        return tensordicts, dict(info)


class VersusPlayCollector(Collector):
    def __init__(
        self,
        env: GomokuEnv,
        policy_black: _policy_t,
        policy_white: _policy_t,
        out_device=None,
        augment: bool = False,
    ):
        """Initializes a collector for versus play data."""
        self._env = env
        self._policy_black = policy_black
        self._policy_white = policy_white
        self._out_device = out_device or self._env.device
        self._augment = augment
        self._t_minus_1 = None
        self._t = None

    def reset(self):
        self._env.reset()
        self._t_minus_1 = None
        self._t = None

    @torch.no_grad()
    def rollout(self, steps: int) -> tuple[TensorDict, TensorDict, dict]:
        """Executes a rollout in the environment (black vs white)."""

        steps = (steps // 2) * 2
        info: defaultdict[str, float] = defaultdict(float)
        self._env.set_post_step(get_log_func(info))
        blacks = []
        whites = []
        start = time.perf_counter()

        if self._t_minus_1 is None and self._t is None:
            self._t_minus_1 = self._env.reset()
            self._t = self._env.reset()
            self._t_minus_1.update(
                {
                    "done": torch.ones(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                }
            )
            with set_interaction_type(type=InteractionType.RANDOM):
                self._t = self._policy_black(self._t)
            self._t.update(
                {
                    "done": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                }
            )

        for i in range(steps // 2):
            (
                transition_black,
                transition_white,
                self._t_minus_1,
                self._t,
            ) = round(
                self._env,
                self._policy_black,
                self._policy_white,
                self._t_minus_1,
                self._t,
            )

            if i == steps // 2 - 1:
                transition_black["next", "done"] = torch.ones(
                    transition_black["next", "done"].shape,
                    dtype=torch.bool,
                    device=transition_black.device,
                )
                transition_white["next", "done"] = torch.ones(
                    transition_white["next", "done"].shape,
                    dtype=torch.bool,
                    device=transition_white.device,
                )

            if self._augment:
                transition_black = augment_transition(transition_black)
                if i != 0:
                    transition_white = augment_transition(transition_white)

            blacks.append(transition_black.to(self._out_device))
            if i != 0:
                whites.append(transition_white.to(self._out_device))

        blacks = torch.stack(blacks, dim=-1) if blacks else None
        whites = torch.stack(whites, dim=-1) if whites else None

        end = time.perf_counter()
        fps = (steps * self._env.num_envs) / (end - start)
        self._env.set_post_step(None)
        info.update({"fps": fps})
        return blacks, whites, dict(info)


class BlackPlayCollector(Collector):
    def __init__(
        self,
        env: GomokuEnv,
        policy_black: _policy_t,
        policy_white: _policy_t,
        out_device=None,
        augment: bool = False,
    ):
        """Collector for capturing game transitions (black player)."""

        self._env = env
        self._policy_black = policy_black
        self._policy_white = policy_white
        self._out_device = out_device or self._env.device
        self._augment = augment
        self._t_minus_1 = None
        self._t = None

    def reset(self):
        self._env.reset()
        self._t_minus_1 = None
        self._t = None

    @torch.no_grad()
    def rollout(self, steps: int) -> tuple[TensorDict, dict]:
        """Executes a data collection session. (black player)"""

        steps = (steps // 2) * 2
        info: defaultdict[str, float] = defaultdict(float)
        self._env.set_post_step(get_log_func(info))
        blacks = []
        start = time.perf_counter()

        if self._t_minus_1 is None and self._t is None:
            self._t_minus_1 = self._env.reset()
            self._t = self._env.reset()
            self._t_minus_1.update(
                {
                    "done": torch.ones(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                }
            )
            with set_interaction_type(type=InteractionType.RANDOM):
                self._t = self._policy_black(self._t)
            self._t.update(
                {
                    "done": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                }
            )

        for i in range(steps // 2):
            (
                transition_black,
                transition_white,
                self._t_minus_1,
                self._t,
            ) = round(
                self._env,
                self._policy_black,
                self._policy_white,
                self._t_minus_1,
                self._t,
                return_black_transitions=True,
                return_white_transitions=False,
            )

            if i == steps // 2 - 1:
                transition_black["next", "done"] = torch.ones(
                    transition_black["next", "done"].shape,
                    dtype=torch.bool,
                    device=transition_black.device,
                )

            if self._augment:
                transition_black = augment_transition(transition_black)

            blacks.append(transition_black.to(self._out_device))

        blacks = torch.stack(blacks, dim=-1) if blacks else None
        end = time.perf_counter()
        fps = (steps * self._env.num_envs) / (end - start)
        self._env.set_post_step(None)
        info.update({"fps": fps})
        return blacks, dict(info)


class WhitePlayCollector(Collector):
    def __init__(
        self,
        env: GomokuEnv,
        policy_black: _policy_t,
        policy_white: _policy_t,
        out_device=None,
        augment: bool = False,
    ):
        """Collector for capturing game transitions (white player)."""

        self._env = env
        self._policy_black = policy_black
        self._policy_white = policy_white
        self._out_device = out_device or self._env.device
        self._augment = augment
        self._t_minus_1 = None
        self._t = None

    def reset(self):
        self._env.reset()
        self._t_minus_1 = None
        self._t = None

    @torch.no_grad()
    def rollout(self, steps: int) -> tuple[TensorDict, dict]:
        """Performs a data collection session. (white player)"""

        steps = (steps // 2) * 2
        info: defaultdict[str, float] = defaultdict(float)
        self._env.set_post_step(get_log_func(info))
        whites = []
        start = time.perf_counter()

        if self._t_minus_1 is None and self._t is None:
            self._t_minus_1 = self._env.reset()
            self._t = self._env.reset()
            self._t_minus_1.update(
                {
                    "done": torch.ones(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "action": -torch.ones(
                        self._env.num_envs,
                        dtype=torch.long,
                        device=self._env.device,
                    ),
                }
            )
            with set_interaction_type(type=InteractionType.RANDOM):
                self._t = self._policy_black(self._t)
            self._t.update(
                {
                    "done": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                }
            )

        for i in range(steps // 2):
            (
                transition_black,
                transition_white,
                self._t_minus_1,
                self._t,
            ) = round(
                self._env,
                self._policy_black,
                self._policy_white,
                self._t_minus_1,
                self._t,
                return_black_transitions=False,
                return_white_transitions=True,
            )

            if i == steps // 2 - 1:
                transition_white["next", "done"] = torch.ones(
                    transition_white["next", "done"].shape,
                    dtype=torch.bool,
                    device=transition_white.device,
                )

            if self._augment:
                if i != 0 and len(transition_white) > 0:
                    transition_white = augment_transition(transition_white)

            if i != 0:
                whites.append(transition_white.to(self._out_device))

        whites = torch.stack(whites, dim=-1) if whites else None
        end = time.perf_counter()
        fps = (steps * self._env.num_envs) / (end - start)
        self._env.set_post_step(None)
        info.update({"fps": fps})
        return whites, dict(info)
