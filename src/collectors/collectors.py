import abc
from collections import defaultdict
import time

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, set_interaction_type, InteractionType

from src.utils.policy import _policy_t
from src.utils.log import get_log_func
from src.envs.gomoku_env import GomokuEnv


# ==============================
# Reward Shaping (Potential 기반)
# ==============================

# Potential weight for our agent
POT_A = 1.0
# Potential weight for opponent
POT_B = 1.0

# True  -> terminal state(게임 종료 시점)에서만 shaping
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


def _max_line_length_board(board: torch.Tensor, player_value: int) -> int:
    """
    board: shape (B, B), 값은 {-1, 0, 1}
    player_value: 1(현재 플레이어), -1(상대 플레이어)
    """
    B = board.shape[0]
    best = 0

    # 1) 가로줄
    for r in range(B):
        line = board[r, :]
        best = max(best, _max_line_length_1d(line, player_value))

    # 2) 세로줄
    for c in range(B):
        line = board[:, c]
        best = max(best, _max_line_length_1d(line, player_value))

    # 3) 대각선 (↘)
    for start in range(B):
        line = board.diagonal(offset=start)
        best = max(best, _max_line_length_1d(line, player_value))
    for start in range(-B + 1, 0):
        line = board.diagonal(offset=start)
        best = max(best, _max_line_length_1d(line, player_value))

    # 4) 반대각선 (↙) = 좌우 flip 후 diagonal
    flipped = torch.flip(board, dims=[1])  # 좌우 반전
    for start in range(B):
        line = flipped.diagonal(offset=start)
        best = max(best, _max_line_length_1d(line, player_value))
    for start in range(-B + 1, 0):
        line = flipped.diagonal(offset=start)
        best = max(best, _max_line_length_1d(line, player_value))

    return best


def _get_local_area(board: torch.Tensor, last_pos: torch.Tensor, radius: int = 4):
    """
    마지막 수(last_pos) 주변의 부분 보드(local board)만 잘라내어 반환.
    board: (B, B)
    last_pos: (2,) = (row, col)
    """
    B = board.shape[0]
    r, c = int(last_pos[0].item()), int(last_pos[1].item())

    r0 = max(0, r - radius)
    r1 = min(B, r + radius + 1)
    c0 = max(0, c - radius)
    c1 = min(B, c + radius + 1)

    return board[r0:r1, c0:c1]


def compute_potential_from_obs(
    obs: torch.Tensor,
    last_action: torch.Tensor | None,
    a: float = POT_A,
    b: float = POT_B,
) -> torch.Tensor:
    """
    Potential Phi(s)를 torch.Tensor(obs) 한 개(단일 환경)에 대해 계산.
    - obs shape: (3, B, B)
      obs[0] = 현재 플레이어의 돌(0/1)
      obs[1] = 상대 플레이어의 돌(0/1)
      obs[2] = 현재 차례의 플레이어(1 or -1)
    - last_action: (2,) = (row, col) 형태의 마지막 수 위치 (없으면 None)
    """
    B = obs.shape[-1]
    my = obs[0]  # (B, B)
    opp = obs[1]  # (B, B)

    # 현재 플레이어의 관점에서 board를 구성
    board = my - opp  # {1, 0, -1}

    # last_action이 없으면 전체 보드 기준
    if last_action is None:
        local = board
    else:
        local = _get_local_area(board, last_action, radius=4)

    my_max = _max_line_length_board(local, player_value=1)
    opp_max = _max_line_length_board(local, player_value=-1)

    return a * float(my_max) - b * float(opp_max)


def compute_potential_from_obs_batch(
    obs: torch.Tensor,
    a: float = POT_A,
    b: float = POT_B,
) -> torch.Tensor:
    """
    로컬(마지막 수 주변 4줄) 기반 potential Phi(s)를 배치 단위로 계산.

    Args:
        obs: (E, 3, B, B)

    Returns:
        phi: (E,) shape의 텐서.
    """
    if obs.dim() != 4:
        raise ValueError(f"obs must have shape (E, 3, B, B), got {obs.shape}")

    E, _, B, _ = obs.shape
    my = obs[:, 0]  # (E, B, B)
    opp = obs[:, 1]  # (E, B, B)
    board = my - opp  # (E, B, B)

    # 각 환경에서 마지막 둔 수를 추정하는 것은 어렵기 때문에,
    # 여기서는 간단히 "보드 전체"를 가지고 Phi를 계산한다고 가정.
    # 필요하다면 last_action 정보를 TensorDict로부터 전달받아
    # 환경마다 로컬 영역만 잘라내는 로직을 추가 가능.

    phi_vals = torch.empty(E, dtype=torch.float32, device=obs.device)
    for e in range(E):
        b_ = board[e]
        my_max = _max_line_length_board(b_, player_value=1)
        opp_max = _max_line_length_board(b_, player_value=-1)
        phi_vals[e] = a * float(my_max) - b * float(opp_max)

    return phi_vals


def compute_shaping_reward(
    tensordict: TensorDict,
    gamma: float,
) -> torch.Tensor:
    """
    Potential 기반 shaping reward r_shaping = gamma * Phi(s') - Phi(s).

    tensordict:
      - "observation": (E, 3, B, B)
      - "next", "observation": (E, 3, B, B)
      - "done": (E,)
      - "next", "done": (E,)

    Returns:
      r_shaping: (E,) 텐서
    """
    obs = tensordict["observation"]
    next_obs = tensordict["next", "observation"]

    # 현재 state와 다음 state의 potential 계산
    phi_s = compute_potential_from_obs_batch(obs)         # (E,)
    phi_s_next = compute_potential_from_obs_batch(next_obs)  # (E,)

    # terminal만 shaping을 주고 싶다면 done 마스크 고려
    if SHAPING_TERMINAL_ONLY:
        done_next = tensordict["next", "done"].float()
        r_shaping = done_next * (gamma * phi_s_next - phi_s)
    else:
        r_shaping = gamma * phi_s_next - phi_s

    return r_shaping


# ==============================
# Transition 생성 및 라운드 로직
# ==============================

def make_transition(
    tensordict: TensorDict,
    reward: torch.Tensor,
    next_tensordict: TensorDict,
    done: torch.Tensor,
    win: torch.Tensor,
) -> TensorDict:
    """
    하나의 transition (s, a, r, s', done, win)을 tensordict 형태로 생성한다.
    - tensordict: 현재 step에서의 상태 및 action 정보
    - next_tensordict: 다음 상태 정보
    """
    t = TensorDict(
        {
            "observation": tensordict["observation"],
            "action": tensordict["action"],
            "reward": reward,
            "done": done,
            "win": win,
            "next": TensorDict(
                {
                    "observation": next_tensordict["observation"],
                    "action_mask": next_tensordict["action_mask"],
                    "done": next_tensordict["done"],
                    "win": next_tensordict["win"],
                },
                batch_size=tensordict.batch_size,
                device=tensordict.device,
            ),
        },
        batch_size=tensordict.batch_size,
        device=tensordict.device,
    )

    if "step_count" in tensordict.keys():
        t.set("step_count", tensordict["step_count"])

    return t


def black_step(env: GomokuEnv, policy: TensorDictModule):
    tensordict = env.black_obs
    tensordict["winning_streak"] = env._winning_streak

    tensordict = policy(tensordict)

    # black이 두기 전 가치 평가 + 한 수 둔 후 가치 평가 차이 -> reward. advantage 형식
    black_values = tensordict.get("state_value", None)
    assert black_values is not None, "Expected 'state_value' in tensordict from policy"

    env.black_obs, reward, done, win = env.black_step(tensordict)
    env.white_obs["win"] = win

    tensordict["second_placed"] = env._second_placed

    # reward + potential_based_shaping
    if (
        not (env._use_potential_reward and env._potential_reward_cv)
        and not env._win_loss_reward
    ):
        if env._use_potential_reward:
            gamma = env._potential_gamma
        else:
            gamma = env._gamma
        # potential-based shaping
        shaped_reward = compute_shaping_reward(
            make_transition(
                tensordict,
                reward,
                env.black_obs,
                done,
                win,
            ),
            gamma=gamma,
        )
        reward = reward + shaped_reward.to(reward.device)

    transition = make_transition(
        tensordict,
        reward,
        env.black_obs,
        done,
        win,
    )

    next_obs = env.black_obs["observation"]
    next_black_values = env._current_black["critic"](
        TensorDict(
            {"observation": next_obs},
            batch_size=tensordict.batch_size,
        ).to(env._current_black["critic"].device)
    )["_states_value"]
    transition["next", "state_value"] = next_black_values

    env._black_done = done

    return transition


def white_step(env: GomokuEnv, policy: TensorDictModule):
    tensordict = env.white_obs
    tensordict["winning_streak"] = env._winning_streak
    tensordict = policy(tensordict)

    white_values = tensordict.get("state_value", None)
    assert white_values is not None, "Expected 'state_value' in tensordict from policy"

    env.white_obs, reward, done, win = env.white_step(tensordict)
    env.black_obs["win"] = win

    tensordict["second_placed"] = env._second_placed

    if (
        not (env._use_potential_reward and env._potential_reward_cv)
        and not env._win_loss_reward
    ):
        if env._use_potential_reward:
            gamma = env._potential_gamma
        else:
            gamma = env._gamma

        shaped_reward = compute_shaping_reward(
            make_transition(
                tensordict,
                reward,
                env.white_obs,
                done,
                win,
            ),
            gamma=gamma,
        )
        reward = reward + shaped_reward.to(reward.device)

    transition = make_transition(
        tensordict,
        reward,
        env.white_obs,
        done,
        win,
    )

    env._white_done = done

    next_obs = env.white_obs["observation"]
    next_white_values = env._current_white["critic"](
        TensorDict(
            {"observation": next_obs},
            batch_size=tensordict.batch_size,
        ).to(env._current_white["critic"].device)
    )["_states_value"]
    transition["next", "state_value"] = next_white_values

    return transition


def round(
    env: GomokuEnv,
    black_policy: TensorDictModule,
    white_policy: TensorDictModule,
    t_minus_1: TensorDict | None = None,
    t: TensorDict | None = None,
    return_black_transitions: bool = True,
    return_white_transitions: bool = True,
) -> tuple[TensorDict | None, TensorDict | None, TensorDict, TensorDict]:
    """
    한 라운드(black 수, white 수)를 진행하고, 각 transition을 만들어서 반환.

    Args:
        env: GomokuEnv 인스턴스
        black_policy: 흑 플레이어 정책
        white_policy: 백 플레이어 정책
        t_minus_1: 이전 transition (없으면 None)
        t: 현재 transition (없으면 None)
        return_black_transitions: True일 때 흑 transition 반환
        return_white_transitions: True일 때 백 transition 반환

    Returns:
        (transition_black, transition_white, t_minus_1, t)
    """
    transition_black = None
    transition_white = None

    if env.whose_turn is None or env.whose_turn == 1:
        # 흑이 두는 차례
        if return_black_transitions:
            transition_black = black_step(env, black_policy)
        else:
            env.black_step(env.black_obs)

        if env.game_over:
            transition_white = None
            t_minus_1 = transition_black
            t = transition_black
            return (transition_black, transition_white, t_minus_1, t)

        if return_white_transitions:
            transition_white = white_step(env, white_policy)

    elif env.whose_turn == -1:
        # 백이 두는 차례
        if return_white_transitions:
            transition_white = white_step(env, white_policy)
        else:
            env.white_step(env.white_obs)

        if env.game_over:
            transition_black = None
            t_minus_1 = transition_white
            t = transition_white
            return (transition_black, transition_white, t_minus_1, t)

        if return_black_transitions:
            transition_black = black_step(env, black_policy)

    # game_over 확인 및 winning streak 갱신
    if env.game_over and env._update_winning_streak:
        winner = env._winner  # 1: black, -1: white, 0: draw 혹은 None
        env.update_winning_streak(winner)

    if env.second_placed:
        if env.whose_turn == 1 and return_black_transitions:
            t_minus_1 = transition_black
            t = transition_white
        elif env.whose_turn == 1 and not return_black_transitions:
            t_minus_1 = transition_white
            t = transition_white
        elif env.whose_turn == -1 and return_black_transitions:
            t_minus_1 = transition_white
            t = transition_black
        elif env.whose_turn == -1 and not return_black_transitions:
            t_minus_1 = transition_black
            t = transition_black

    return transition_black, transition_white, t_minus_1, t


def self_play_step(
    env: GomokuEnv,
    policy: TensorDictModule,
    t_minus_1: TensorDict | None = None,
    t: TensorDict | None = None,
) -> tuple[TensorDict, TensorDict | None, TensorDict | None]:
    """
    self-play 환경에서 한 수를 진행하고 transition 생성.

    Args:
        env: GomokuEnv 인스턴스
        policy: 두 플레이어가 공유하는 동일한 policy
        t_minus_1: 이전 transition
        t: 현재 transition

    Returns:
        (transition, t_minus_1, t)
    """
    if env.whose_turn == 1:
        transition = black_step(env, policy)
    else:
        transition = white_step(env, policy)

    # game_over 확인 및 winning streak 갱신
    if env.game_over and env._update_winning_streak:
        winner = env._winner
        env.update_winning_streak(winner)

    if env.second_placed:
        if env.whose_turn == 1:
            t_minus_1 = transition
            t = transition
        elif env.whose_turn == -1:
            t_minus_1 = transition
            t = transition

    return transition, t_minus_1, t


class Collector(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def rollout(self, steps: int) -> tuple[TensorDict, dict]:
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
    ):
        """Initializes a collector for self-play data."""
        self._env = env
        self._policy = policy
        self._out_device = out_device or self._env.device
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
            self._t_minus_1 = TensorDict(
                {
                    "observation": self._env.black_obs["observation"].clone(),
                    "action": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.long,
                        device=self._env.device,
                    ),
                    "action_mask": torch.ones(
                        self._env.num_envs,
                        self._env.board_size * self._env.board_size,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                    "reward": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.float32,
                        device=self._env.device,
                    ),
                    "done": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                    "win": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                    "step_count": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.int64,
                        device=self._env.device,
                    ),
                },
                batch_size=self._env.num_envs,
                device=self._env.device,
            )
            self._t = TensorDict(
                {
                    "observation": self._env.white_obs["observation"].clone(),
                    "action": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.long,
                        device=self._env.device,
                    ),
                    "action_mask": torch.ones(
                        self._env.num_envs,
                        self._env.board_size * self._env.board_size,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                    "reward": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.float32,
                        device=self._env.device,
                    ),
                    "done": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                    "win": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                    "step_count": torch.ones(
                        self._env.num_envs,
                        dtype=torch.int64,
                        device=self._env.device,
                    ),
                },
                batch_size=self._env.num_envs,
                device=self._env.device,
            )

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
    ):
        """Initializes a collector for versus play data."""
        self._env = env
        self._policy_black = policy_black
        self._policy_white = policy_white
        self._out_device = out_device or self._env.device
        self._t_minus_1 = None
        self._t = None

    def reset(self):
        self._env.reset()
        self._t_minus_1 = None
        self._t = None

    @torch.no_grad()
    def rollout(self, steps: int) -> tuple[TensorDict, TensorDict, dict]:
        """Executes a versus-play rollout."""
        steps = (steps // 2) * 2

        info: defaultdict[str, float] = defaultdict(float)
        self._env.set_post_step(get_log_func(info))
        blacks = []
        whites = []
        start = time.perf_counter()

        if self._t_minus_1 is None and self._t is None:
            self._t_minus_1 = TensorDict(
                {
                    "observation": self._env.black_obs["observation"].clone(),
                    "action": torch.zeros(
                        self._env.num_envs, dtype=torch.long, device=self._env.device
                    ),
                    "action_mask": torch.ones(
                        self._env.num_envs,
                        self._env.board_size * self._env.board_size,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                    "reward": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.float32,
                        device=self._env.device,
                    ),
                    "done": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "step_count": torch.zeros(
                        self._env.num_envs, dtype=torch.int64, device=self._env.device
                    ),
                },
                batch_size=self._env.num_envs,
                device=self._env.device,
            )
            self._t = TensorDict(
                {
                    "observation": self._env.white_obs["observation"].clone(),
                    "action": torch.zeros(
                        self._env.num_envs, dtype=torch.long, device=self._env.device
                    ),
                    "action_mask": torch.ones(
                        self._env.num_envs,
                        self._env.board_size * self._env.board_size,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                    "reward": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.float32,
                        device=self._env.device,
                    ),
                    "done": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "step_count": torch.ones(
                        self._env.num_envs, dtype=torch.int64, device=self._env.device
                    ),
                },
                batch_size=self._env.num_envs,
                device=self._env.device,
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
    ):
        """Collector for capturing game transitions (black player)."""

        self._env = env
        self._policy_black = policy_black
        self._policy_white = policy_white
        self._out_device = out_device or self._env.device
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
            self._t_minus_1 = TensorDict(
                {
                    "observation": self._env.black_obs["observation"].clone(),
                    "action": torch.zeros(
                        self._env.num_envs, dtype=torch.long, device=self._env.device
                    ),
                    "action_mask": torch.ones(
                        self._env.num_envs,
                        self._env.board_size * self._env.board_size,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                    "reward": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.float32,
                        device=self._env.device,
                    ),
                    "done": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "step_count": torch.zeros(
                        self._env.num_envs, dtype=torch.int64, device=self._env.device
                    ),
                },
                batch_size=self._env.num_envs,
                device=self._env.device,
            )
            self._t = TensorDict(
                {
                    "observation": self._env.white_obs["observation"].clone(),
                    "action": torch.zeros(
                        self._env.num_envs, dtype=torch.long, device=self._env.device
                    ),
                    "action_mask": torch.ones(
                        self._env.num_envs,
                        self._env.board_size * self._env.board_size,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                    "reward": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.float32,
                        device=self._env.device,
                    ),
                    "done": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "step_count": torch.ones(
                        self._env.num_envs, dtype=torch.int64, device=self._env.device
                    ),
                },
                batch_size=self._env.num_envs,
                device=self._env.device,
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
    ):
        """Collector for capturing game transitions (white player)."""

        self._env = env
        self._policy_black = policy_black
        self._policy_white = policy_white
        self._out_device = out_device or self._env.device
        self._t_minus_1 = None
        self._t = None

    def reset(self):
        self._env.reset()
        self._t_minus_1 = None
        self._t = None

    @torch.no_grad()
    def rollout(self, steps: int) -> tuple[TensorDict, dict]:
        """Executes a data collection session. (white player)"""

        steps = (steps // 2) * 2
        info: defaultdict[str, float] = defaultdict(float)
        self._env.set_post_step(get_log_func(info))

        whites = []
        start = time.perf_counter()

        if self._t_minus_1 is None and self._t is None:
            self._t_minus_1 = TensorDict(
                {
                    "observation": self._env.black_obs["observation"].clone(),
                    "action": torch.zeros(
                        self._env.num_envs, dtype=torch.long, device=self._env.device
                    ),
                    "action_mask": torch.ones(
                        self._env.num_envs,
                        self._env.board_size * self._env.board_size,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                    "reward": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.float32,
                        device=self._env.device,
                    ),
                    "done": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "step_count": torch.zeros(
                        self._env.num_envs, dtype=torch.int64, device=self._env.device
                    ),
                },
                batch_size=self._env.num_envs,
                device=self._env.device,
            )
            self._t = TensorDict(
                {
                    "observation": self._env.white_obs["observation"].clone(),
                    "action": torch.zeros(
                        self._env.num_envs, dtype=torch.long, device=self._env.device
                    ),
                    "action_mask": torch.ones(
                        self._env.num_envs,
                        self._env.board_size * self._env.board_size,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                    "reward": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.float32,
                        device=self._env.device,
                    ),
                    "done": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "win": torch.zeros(
                        self._env.num_envs, dtype=torch.bool, device=self._env.device
                    ),
                    "step_count": torch.ones(
                        self._env.num_envs, dtype=torch.int64, device=self._env.device
                    ),
                },
                batch_size=self._env.num_envs,
                device=self._env.device,
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

            if i != 0:
                whites.append(transition_white.to(self._out_device))

        whites = torch.stack(whites, dim=-1) if whites else None
        end = time.perf_counter()
        fps = (steps * self._env.num_envs) / (end - start)
        self._env.set_post_step(None)
        info.update({"fps": fps})
        return whites, dict(info)
