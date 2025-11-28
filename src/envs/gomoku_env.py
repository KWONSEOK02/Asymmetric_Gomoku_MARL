from typing import Callable
from tensordict import TensorDict
import torch
import torch.nn.functional as F

from torchrl.data.tensor_specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    BinaryDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)

from src.utils.policy import _policy_t  # 호환성 유지용 import (직접 사용하진 않음)
from src.utils.misc import add_prefix
from src.utils.log import get_log_func
from src.envs.core import Gomoku


class GomokuEnv:
    def __init__(
        self,
        num_envs: int,
        board_size: int,
        device=None,
    ):
        """Parallel Gomoku environment with reward shaping."""
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
            shape=[num_envs],
            device=self.device,
        )

        self.action_spec = DiscreteTensorSpec(
            board_size * board_size,
            shape=[num_envs],
            device=self.device,
        )

        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=[num_envs, 1],
            device=self.device,
        )

        self._post_step: Callable[[TensorDict], None] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Helper: max line length (<=5) via conv (GPU friendly)
    # ------------------------------------------------------------------
    def _max_line_in_five(self, mask: torch.Tensor) -> torch.Tensor:
        """Return max number of stones in any 5-long window (horizontal/vertical/diag).

        Args:
            mask: (E,B,B) bool or {0,1} float tensor indicating occupied cells.
        """
        if mask.dtype not in (torch.float32, torch.float64):
            x = mask.float().unsqueeze(1)  # (E,1,B,B)
        else:
            x = mask.unsqueeze(1)

        g = self.gomoku
        out_h = F.conv2d(x, g.kernel_horizontal)  # (E,1,B-4,B)
        out_v = F.conv2d(x, g.kernel_vertical)    # (E,1,B,B-4)
        out_d = F.conv2d(x, g.kernel_diagonal)    # (E,2,B-4,B-4)

        max_h = out_h.flatten(start_dim=1).amax(dim=1)
        max_v = out_v.flatten(start_dim=1).amax(dim=1)
        max_d = out_d.flatten(start_dim=1).amax(dim=1)

        return torch.stack([max_h, max_v, max_d], dim=1).amax(dim=1)

    # ------------------------------------------------------------------
    # Helper: local threat-3 / threat-4 block detection around the move
    # ------------------------------------------------------------------
    def _compute_block_threats_local(
        self,
        board_before: torch.Tensor,   # (E,B,B)
        action: torch.Tensor,         # (E,)
        opp_piece: torch.Tensor,      # (E,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Count how many threat-3 / threat-4 are blocked by this move (local check).

        - Threat-3 block:
            상대가 가로나 세로 / 대각선 한쪽 방향으로 3개 연속으로 두어둔 줄을,
            그 줄의 끝 칸에 둬서 차단했을 때.
        - Threat-4 block:
            상대가 한쪽 방향으로 4개 연속으로 두어둔 줄(오목 직전)을,
            그 줄의 끝 칸에 둬서 차단했을 때.

        action이 놓인 칸을 기준으로 가로/세로/두 대각선 4줄만 검사하므로
        복잡도는 O(num_envs * board_size) 수준입니다.
        """
        E, B, _ = board_before.shape
        block3 = torch.zeros(E, device=self.device, dtype=torch.float32)
        block4 = torch.zeros(E, device=self.device, dtype=torch.float32)

        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for e in range(E):
            a = int(action[e].item())
            if a < 0:
                continue  # env_mask로 이미 필터링되는 게 정상이고, 여기선 방어용
            x = a // B
            y = a % B
            if x < 0 or x >= B or y < 0 or y >= B:
                continue

            opp = int(opp_piece[e].item())

            for dx, dy in dirs:
                # + 방향
                cnt_pos = 0
                ix, iy = x + dx, y + dy
                while 0 <= ix < B and 0 <= iy < B and int(board_before[e, ix, iy].item()) == opp:
                    cnt_pos += 1
                    ix += dx
                    iy += dy

                # - 방향
                cnt_neg = 0
                ix, iy = x - dx, y - dy
                while 0 <= ix < B and 0 <= iy < B and int(board_before[e, ix, iy].item()) == opp:
                    cnt_neg += 1
                    ix -= dx
                    iy -= dy

                # 우리 수가 들어오는 칸은 이전에는 항상 0이므로
                # 상대 연속 줄은 한쪽 방향으로만 존재
                max_run = max(cnt_pos, cnt_neg)
                if max_run >= 4:
                    block4[e] += 1.0
                elif max_run == 3:
                    block3[e] += 1.0

        return block3, block4

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
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

    def step(self, tensordict: TensorDict) -> TensorDict:
        action: torch.Tensor = tensordict.get("action")
        env_mask: torch.Tensor | None = tensordict.get("env_mask", None)

        if env_mask is None:
            env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # ----- step 이전 상태 백업 (보상 계산용) -----
        board_before = self.gomoku.board.clone()      # (E,B,B)
        turn_before = self.gomoku.turn.clone()        # (E,)
        move_count_before = self.gomoku.move_count.clone()
        episode_len = move_count_before + 1

        # 현재 수를 두는 플레이어 (+1 / -1), 상대는 -piece
        piece = torch.where(
            turn_before == 0,
            torch.ones_like(turn_before, dtype=torch.long),
            -torch.ones_like(turn_before, dtype=torch.long),
        )  # (E,)
        opp_piece = -piece

        # threat-3 / threat-4 차단 개수 (local)
        block3_count, block4_count = self._compute_block_threats_local(
            board_before, action, opp_piece
        )

        # ----- 실제 수 두기 -----
        win, illegal = self.gomoku.step(action=action, env_mask=env_mask)
        assert not illegal.any()

        done = win
        black_win = win & (episode_len % 2 == 1)
        white_win = win & (episode_len % 2 == 0)

        board_after = self.gomoku.board

        # ----- 자기/상대 최장 줄 길이 변화 (GPU conv 기반) -----
        piece_view = piece.view(-1, 1, 1)
        own_before = board_before == piece_view
        own_after = board_after == piece_view
        opp_before = board_before == -piece_view
        opp_after = board_after == -piece_view

        own_max_before = self._max_line_in_five(own_before)
        own_max_after = self._max_line_in_five(own_after)
        opp_max_before = self._max_line_in_five(opp_before)
        opp_max_after = self._max_line_in_five(opp_after)

        delta_self_len = (own_max_after - own_max_before).clamp(min=0.0)
        delta_block_len = (opp_max_before - opp_max_after).clamp(min=0.0)

        # ----- 중앙 선호 -----
        B = self.board_size
        x = action // B
        y = action % B
        center = (B - 1) / 2.0
        dist_center = (x.float() - center).abs() + (y.float() - center).abs()
        max_dist = 2.0 * (B - 1)
        center_reward = -0.03 * (dist_center / max_dist)

        # ----- 타임 페널티 -----
        step_penalty = -0.002

        # 실제로 수를 둔 env에만 shaping 반영
        valid_mask = env_mask & (~illegal)

        # ----- 하이퍼파라미터 (필요 시 조정) -----
        w_len_self = 0.04   # 자기 줄 늘리기
        w_len_block = 0.06  # 상대 줄 길이 차단
        w_block3 = 0.08     # threat-3 차단
        w_block4 = 0.20     # threat-4 차단

        shaping = (
            w_len_self * delta_self_len
            + w_len_block * delta_block_len
            + w_block3 * block3_count
            + w_block4 * block4_count
            + center_reward
            + step_penalty
        )
        shaping = shaping * valid_mask.float()

        # 승리 보상 (+1)
        win_reward = win.float()

        reward = (win_reward + shaping).unsqueeze(-1)

        out_td = TensorDict({}, self.batch_size, device=self.device)
        out_td.update(
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
                    "block3_count": block3_count,
                    "block4_count": block4_count,
                },
            }
        )

        if self._post_step:
            self._post_step(out_td)

        return out_td

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
