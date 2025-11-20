import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
from tensordict import TensorDict
from torchrl.data.tensor_specs import (
    DiscreteTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    BinaryDiscreteTensorSpec,
)

# 만든 모듈들 임포트
from src.evaluation.evaluator import evaluate_win_rate
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


@hydra.main(version_base=None, config_path="../configs", config_name="eval_config")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 1. 스펙 생성
    action_spec, observation_spec = _build_specs(cfg)

    # 2. 정책 로드
    logger.info(f"Loading Black Policy: {cfg.algo_black.name}")
    black_policy = _load_policy(
        cfg.black_checkpoint, cfg.algo_black, 
        action_spec, observation_spec, cfg.device
    )

    logger.info(f"Loading White Policy: {cfg.algo_white.name}")
    white_policy = _load_policy(
        cfg.white_checkpoint, cfg.algo_white, 
        action_spec, observation_spec, cfg.device
    )

    # 3. 평가 실행
    logger.info(f"Starting evaluation on {cfg.num_eval_envs} games...")
    
    results = evaluate_win_rate(
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
    
    # 팀원이 요청한 '누적 보상' (총 승리 횟수)
    logger.info(f"Cumulative Reward (Total Wins): {results['cumulative_reward']} / {cfg.num_eval_envs}")
    
    # 팀원이 요청한 '평균 보상' (승률과 동일)
    logger.info(f"Average Reward (Win Prob):    {results['average_reward']:.4f}")
    
    # 직관적인 승률 표시
    logger.info(f"Win Rate (Black):             {results['win_rate']}")
    logger.info("=" * 30)

if __name__ == "__main__":
    main()