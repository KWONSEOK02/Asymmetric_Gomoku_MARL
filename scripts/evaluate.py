import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch

import warnings
warnings.filterwarnings("ignore")

from tensordict import TensorDict
from torchrl.data.tensor_specs import (
    DiscreteTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    BinaryDiscreteTensorSpec,
)

from src.policy import get_policy

# --- model_duel.py와 동일한 헬퍼 함수 ---
def _build_specs(cfg: DictConfig):
    board_size = cfg.board_size
    device = cfg.device
    num_envs = cfg.num_eval_envs # 평가용 env 개수

    action_spec = DiscreteTensorSpec(
        board_size * board_size,
        shape=[num_envs],
        device=device,
    )
    observation_spec = CompositeSpec(
        {
            "observation": UnboundedContinuousTensorSpec(
                device=device,
                shape=[num_envs, 3, board_size, board_size],
            ),
            "action_mask": BinaryDiscreteTensorSpec(
                n=board_size * board_size,
                device=device,
                shape=[num_envs, board_size * board_size],
                dtype=torch.bool,
            ),
        },
        shape=[num_envs],
        device=device,
    )
    return action_spec, observation_spec

def _load_policy(checkpoint_path, algo_cfg, action_spec, observation_spec, device):
    policy = get_policy(
        name=algo_cfg.name,
        cfg=algo_cfg,
        action_spec=action_spec,
        observation_spec=observation_spec,
        device=device,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy

# =============================================================================
# [수정] evaluate_win_rate 함수 재정의 (랜덤 오프닝 추가를 위해 여기로 가져옴)
# =============================================================================
from src.envs.gomoku_env import GomokuEnv

def evaluate_win_rate_random_opening(
    black_policy,
    white_policy,
    board_size: int = 15,
    num_envs: int = 100,
    device: str = "cuda",
):
    black_policy.eval()
    white_policy.eval()
    env = GomokuEnv(num_envs=num_envs, board_size=board_size, device=device)
    td = env.reset()


    if "done" in td.keys():
        done = td.get("done")
    else:
        done = torch.zeros(num_envs, dtype=torch.bool, device=device)
    first_step = True
    with torch.no_grad():
        while not done.all():
            turn = env.gomoku.turn
        
            # 첫 번째 수(흑돌)일 경우, AI 정책 대신 5x5 랜덤 오프닝 적용
            if first_step:
                # 중앙 좌표 계산
                center = board_size // 2
                margin = 2  # 중앙 기준 +-2칸 (총 5칸: center-2 ~ center+2)

                # 범위 내 랜덤 좌표 생성 (High는 exclusive하므로 +1)
                # shape: [num_envs]
                rand_x = torch.randint(center - margin, center + margin + 1, (num_envs,), device=device)
                rand_y = torch.randint(center - margin, center + margin + 1, (num_envs,), device=device)

                # 좌표를 액션 인덱스로 변환 (x * board_size + y)
                actions = rand_x * board_size + rand_y

                # 흑돌(0번 턴)의 행동으로 간주
                # (백돌이 먼저 두는 경우는 없다고 가정)

            else:
                # 기존 로직 (2수째부터는 AI가 둠)
                policy_input = td.select("observation", "action_mask")
                td_black = black_policy(policy_input.clone())
                td_white = white_policy(policy_input.clone())
                actions_black = td_black["action"]
                actions_white = td_white["action"]
                actions = torch.where(turn == 0, actions_black, actions_white)
                if "action" in td.keys():
                    prev_action = td["action"]
                    actions = torch.where(done, prev_action, actions)
            td.set("action", actions)
            td = env.step(td)
            done = td.get("done")
            first_step = False # 플래그 해제
    stats = td.get("stats", None)
    if stats is None:
        raise RuntimeError("stats missing")
    black_wins = stats["black_win"].sum().item()
    white_wins = stats["white_win"].sum().item()
    draws = num_envs - black_wins - white_wins
    total_wins_black = float(black_wins)
    average_reward_black = total_wins_black / float(num_envs)
    return {
        "total_wins": total_wins_black,
        "average_reward": average_reward_black,
        "win_rate": f"{average_reward_black * 100:.1f}%",
        "black_wins": int(black_wins),
        "white_wins": int(white_wins),
        "draws": int(draws),
    }

@hydra.main(version_base=None, config_path="../configs", config_name="eval_config")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 1. 스펙 생성
    action_spec, observation_spec = _build_specs(cfg)

    # 2. 정책 로드
    logger.info(f"Loading Black Policy: {cfg.algo.name}")
    black_policy = _load_policy(
        cfg.black_checkpoint, cfg.algo,
        action_spec, observation_spec, cfg.device
    )

    logger.info(f"Loading White Policy: {cfg.algo.name}")
    white_policy = _load_policy(
        cfg.white_checkpoint, cfg.algo,
        action_spec, observation_spec, cfg.device
    )

    # 3. 평가 실행
    logger.info(f"Starting evaluation on {cfg.num_eval_envs} games...")
    logger.info(f"Mode: 5x5 Center Random Opening Enabled") # [수정] 로그 추가

    # [수정] 수정한 함수 호출
    results = evaluate_win_rate_random_opening(
        black_policy,
        white_policy,
        board_size=cfg.board_size,
        num_envs=cfg.num_eval_envs,
        device=cfg.device
    )

    # 4. 결과 출력 (누적 보상, 평균 보상 추가)
    logger.info("=" * 30)
    logger.info("      EVALUATION RESULTS      ")
    logger.info("=" * 30)

    # '누적 보상' (총 승리 횟수)
    logger.info(f"Total Wins (Black):             {results['total_wins']} / {cfg.num_eval_envs}")
    logger.info(f"Total Wins (White):             {results['white_wins']}")
    logger.info(f"Draws:                          {results['draws']}")
    # '평균 보상' (승률과 동일)
    logger.info(f"Average Reward (Win Prob):    {results['average_reward']:.4f}")

    # 승률 표시
    logger.info(f"Win Rate (Black):             {results['win_rate']}")
    logger.info("=" * 30)

if __name__ == "__main__":
    main()
