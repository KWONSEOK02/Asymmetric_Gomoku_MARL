from typing import Union, Callable
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, set_interaction_type, InteractionType
import torch
import torch.nn.functional as F

from torchrl.data.tensor_specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    BinaryDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
import time

from src.utils.policy import _policy_t
from src.utils.misc import add_prefix
from src.utils.log import get_log_func
from src.envs.core import Gomoku

from collections import defaultdict


class GomokuEnv:
    def __init__(
        self,
        num_envs: int,
        board_size: int,
        device=None,
    ):
        """Initializes a parallel Gomoku environment."""
        self.gomoku = Gomoku(
            num_envs=num_envs, board_size=board_size, device=device
        )

        self.observation_spec = CompositeSpec(
            {
                "observation": UnboundedContinuousTensorSpec(
                    device=self.device,
                    shape=[num_envs, 3, board_size, board_size],
                ),
                "action_mask": BinaryDiscreteTensorSpec(
                    n=board_size * board_size,
                    device=self.device,
                    shape=[num_envs, board_size * board_size],
                    dtype=torch.bool,
                ),
            },
            shape=[
                num_envs,
            ],
            device=self.device,
        )
        self.action_spec = DiscreteTensorSpec(
            board_size * board_size,
            shape=[
                num_envs,
            ],
            device=self.device,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=[num_envs, 1],
            device=self.device,
        )

        self._post_step: Callable[
            [
                TensorDict,
            ],
            None,
        ] | None = None

    @property
    def batch_size(self):
        return torch.Size((self.num_envs,))

    @property
    def board_size(self):
        return self.gomoku.board_size

    @property
    def device(self):
        return self.gomoku.device

    @property
    def num_envs(self):
        return self.gomoku.num_envs

    def _max_line_in_five(self, mask: torch.Tensor) -> torch.Tensor:
        """각 env에서 5칸 윈도우 기준 최대 연속 돌 개수(<=5)를 리턴."""
        if mask.dtype not in (torch.float32, torch.float64):
            x = mask.float().unsqueeze(1)  # (E,1,B,B)
        else:
            x = mask.unsqueeze(1)

        gomoku = self.gomoku

        out_h = F.conv2d(x, gomoku.kernel_horizontal)  # (E,1,B-4,B)
        out_v = F.conv2d(x, gomoku.kernel_vertical)    # (E,1,B,B-4)
        out_d = F.conv2d(x, gomoku.kernel_diagonal)    # (E,2,B-4,B-4)

        max_h = out_h.flatten(start_dim=1).amax(dim=1)  # (E,)
        max_v = out_v.flatten(start_dim=1).amax(dim=1)  # (E,)
        max_d = out_d.flatten(start_dim=1).amax(dim=1)  # (E,)

        return torch.stack([max_h, max_v, max_d], dim=1).amax(dim=1)

    @staticmethod
    def _count_threats_single(board_2d: torch.Tensor, opp_val: int) -> tuple[int, int]:
        """
        단일 보드(2D)에 대해 상대의 threat-3 / threat-4 개수를 센다.

        - Threat-3: 열린 3목 패턴 0 1 1 1 0
        - Threat-4: 한쪽만 막힌 4목 패턴
            -1 1 1 1 1 0   또는   0 1 1 1 1 -1

        여기서: 1 = 상대 돌, 0 = 빈칸, -1 = 내 돌 또는 보드 밖(엣지).
        """
        device = board_2d.device
        B = board_2d.shape[0]

        threat3 = 0
        threat4 = 0

        def process_line(line: torch.Tensor):
            nonlocal threat3, threat4
            # 보드 값을 {-1, 0, 1}로 매핑
            vals = torch.full_like(line, fill_value=-1)
            vals[line == opp_val] = 1
            vals[line == 0] = 0

            pad = torch.tensor([-1], dtype=vals.dtype, device=device)
            v = torch.cat([pad, vals, pad], dim=0)  # (L+2,)
            L = v.shape[0]

            # Threat-3: 0 1 1 1 0 (양 끝이 빈칸인 열린 3목)
            if L >= 5:
                for i in range(L - 4):
                    w = v[i: i + 5]
                    if (
                        int(w[0].item()) == 0
                        and int(w[1].item()) == 1
                        and int(w[2].item()) == 1
                        and int(w[3].item()) == 1
                        and int(w[4].item()) == 0
                    ):
                        threat3 += 1

            # Threat-4: -1 1 1 1 1 0  또는  0 1 1 1 1 -1
            if L >= 6:
                for i in range(L - 5):
                    w = v[i: i + 6]
                    center = w[1:5]
                    if bool((center == 1).all()):
                        first = int(w[0].item())
                        last = int(w[5].item())
                        if (first == -1 and last == 0) or (first == 0 and last == -1):
                            threat4 += 1

        # 가로
        for r in range(B):
            process_line(board_2d[r, :])
        # 세로
        for c in range(B):
            process_line(board_2d[:, c])
        # 주대각선들
        for offset in range(-B + 5, B - 4):
            diag = board_2d.diagonal(offset=offset)
            if diag.numel() >= 5:
                process_line(diag)
        # 역대각선들
        flipped = torch.flip(board_2d, dims=[1])
        for offset in range(-B + 5, B - 4):
            diag = flipped.diagonal(offset=offset)
            if diag.numel() >= 5:
                process_line(diag)

        return threat3, threat4

    def _count_threats(
        self, board: torch.Tensor, opp_piece: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        배치 보드에 대해 threat-3 / threat-4 개수를 센다.

        Args:
            board: (E,B,B) 보드 텐서 (0, 1, -1)
            opp_piece: (E,) 상대 돌 값 (+1 또는 -1)

        Returns:
            threat3, threat4: 각 (E,) float 텐서
        """
        E, B, _ = board.shape
        threat3 = torch.zeros(E, device=self.device, dtype=torch.float32)
        threat4 = torch.zeros(E, device=self.device, dtype=torch.float32)

        for e in range(E):
            t3, t4 = self._count_threats_single(
                board[e], int(opp_piece[e].item())
            )
            threat3[e] = float(t3)
            threat4[e] = float(t4)

        return threat3, threat4

    def reset(self, env_indices: torch.Tensor | None = None) -> TensorDict:
        self.gomoku.reset(env_indices=env_indices)
        tensordict = TensorDict(
            {
                "observation": self.gomoku.get_encoded_board(),
                "action_mask": self.gomoku.get_action_mask(),
            },
            self.batch_size,
            device=self.device,
        )
        return tensordict

    def step(
        self,
        tensordict: TensorDict,
    ) -> TensorDict:
        action: torch.Tensor = tensordict.get("action")
        env_mask: torch.Tensor | None = tensordict.get("env_mask", None)

        # --- 보상 쉐이핑을 위한 step 이전 상태 백업 ---
        board_before = self.gomoku.board.clone()
        turn_before = self.gomoku.turn.clone()
        move_count_before = self.gomoku.move_count.clone()

        episode_len = move_count_before + 1

        # 현재 수를 두는 플레이어 돌 값 (+1 / -1), 상대는 -piece
        piece = torch.where(
            turn_before == 0,
            torch.ones_like(turn_before, dtype=torch.long),
            -torch.ones_like(turn_before, dtype=torch.long),
        )  # (E,)
        opp_piece = -piece  # (E,)

        # 수 두기 전 상대 threat-3 / threat-4 개수
        opp_threat3_before, opp_threat4_before = self._count_threats(
            board_before, opp_piece
        )

        # --- 실제 수 두기 ---
        win, illegal = self.gomoku.step(action=action, env_mask=env_mask)

        # action_mask를 잘 따르면 illegal은 발생하지 않는 게 정상
        assert not illegal.any()

        done = win
        black_win = win & (episode_len % 2 == 1)
        white_win = win & (episode_len % 2 == 0)

        board_after = self.gomoku.board

        # 수 둔 후 상대 threat-3 / threat-4 개수
        opp_threat3_after, opp_threat4_after = self._count_threats(
            board_after, opp_piece
        )

        # 이번 수로 차단한 threat 개수
        delta_block3 = (opp_threat3_before - opp_threat3_after).clamp(min=0.0)
        delta_block4 = (opp_threat4_before - opp_threat4_after).clamp(min=0.0)

        # --- 자기/상대 줄 길이 기반 쉐이핑 ---
        piece_view = piece.view(-1, 1, 1)
        own_before = board_before == piece_view
        own_after = board_after == piece_view
        opp_before = board_before == (-piece_view)
        opp_after = board_after == (-piece_view)

        own_max_before = self._max_line_in_five(own_before)
        own_max_after = self._max_line_in_five(own_after)
        opp_max_before = self._max_line_in_five(opp_before)
        opp_max_after = self._max_line_in_five(opp_after)

        delta_self_len = (own_max_after - own_max_before).clamp(min=0.0)
        delta_block_len = (opp_max_before - opp_max_after).clamp(min=0.0)

        # --- 중앙 선호(작은 보너스) ---
        board_size = self.board_size
        x = action // board_size
        y = action % board_size
        center = (board_size - 1) / 2.0
        dist_center = (x.float() - center).abs() + (y.float() - center).abs()
        max_dist = 2.0 * (board_size - 1)
        center_reward = -0.03 * (dist_center / max_dist)

        # --- 타임 패널티(매 수당 작은 페널티) ---
        step_penalty = -0.002

        # 실제로 수를 둔 env만 보상 반영
        if env_mask is None:
            valid_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            valid_mask = env_mask & (~illegal)

        # 쉐이핑 가중치
        w_len_self = 0.04   # 자기 줄 늘리기
        w_len_block = 0.06  # 상대 줄 차단
        w_block3 = 0.08     # threat-3 차단 보상
        w_block4 = 0.20     # threat-4(오목 직전) 차단 보상

        shaping = (
            w_len_self * delta_self_len
            + w_len_block * delta_block_len
            + w_block3 * delta_block3
            + w_block4 * delta_block4
            + center_reward
            + step_penalty
        )

        shaping = shaping * valid_mask.float()

        # 승리 보상 (+1), 패배는 여기서 0, 패널티/형태 보상은 shaping에 포함
        win_reward = win.float()

        reward = (win_reward + shaping).unsqueeze(-1)  # (E,1)

        tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict.update(
            {
                "observation": self.gomoku.get_encoded_board(),
                "action_mask": self.gomoku.get_action_mask(),
                "reward": reward,
                "done": done,
                "win": win,
                "stats": {
                    "episode_len": episode_len,
                    "black_win": black_win,
                    "white_win": white_win,
                    "delta_self_len": delta_self_len,
                    "delta_block_len": delta_block_len,
                    "delta_block3": delta_block3,
                    "delta_block4": delta_block4,
                },
            }
        )
        if self._post_step:
            self._post_step(tensordict)
        return tensordict

    def step_and_maybe_reset(
        self,
        tensordict: TensorDict,
        env_mask: torch.Tensor | None = None,
    ) -> TensorDict:
        if env_mask is not None:
            tensordict.set("env_mask", env_mask)
        next_tensordict = self.step(tensordict=tensordict)
        tensordict.exclude("env_mask", inplace=True)

        done: torch.Tensor = next_tensordict.get("done")
        env_ids = done.nonzero().squeeze(0)
        reset_td = self.reset(env_indices=env_ids)
        next_tensordict.update(reset_td)
        return next_tensordict

    def set_post_step(self, post_step: Callable[[TensorDict], None] | None = None):
        self._post_step = post_step
