import torch
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from datetime import datetime

from src.envs import GomokuEnv
from src.utils.policy import uniform_policy
from src.utils.misc import set_seed
from src.collectors.collectors import (
    BlackPlayCollector,
    WhitePlayCollector,
    SelfPlayCollector,
    VersusPlayCollector,
)
from src.policy import get_policy, get_pretrained_policy

# 로거 설정
log = logging.getLogger(__name__)

# 실행 시마다 고유한 타임스탬프 폴더 이름
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logger(log_dir: str):
    """
    log_dir 아래에 train.log를 만들고,
    콘솔 + 파일 둘 다에 로그가 찍히도록 설정.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 이전 핸들러(기존 Hydra / 기본 설정 등) 제거해서 중복 출력 방지
    logger.handlers.clear()

    # 공통 포맷
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 파일 핸들러 (train.log)
    log_path = os.path.join(log_dir, "train.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 콘솔 핸들러
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def run_versus_play_training(cfg: DictConfig, env: GomokuEnv, run_dir: str):
    """
    VersusPlayCollector를 사용해서 흑/백 두 에이전트를 동시에 학습시키는 루프.
    _refresh_collector_models()나 update_policies()는 사용하지 않음.
    """
    log = logging.getLogger(__name__)
    log.info("--- VersusPlay Training ---")

    # 1) 흑 / 백 에이전트 각각 생성 (같은 algo 설정을 공유)
    log.info(f"Creating Black Agent ({cfg.algo.name})")
    agent_black = get_policy(
        name=cfg.algo.name,
        cfg=cfg.algo,
        action_spec=env.action_spec,
        observation_spec=env.observation_spec,
        device=cfg.device,
    )

    log.info(f"Creating White Agent ({cfg.algo.name})")
    agent_white = get_policy(
        name=cfg.algo.name,
        cfg=cfg.algo,
        action_spec=env.action_spec,
        observation_spec=env.observation_spec,
        device=cfg.device,
    )

    # 2) 이어서 학습하는 경우 각각 로드 (원하면)
    if cfg.get("train_checkpoint_black"):
        log.info(f"Loading black agent checkpoint from {cfg.train_checkpoint_black}")
        state_dict = torch.load(cfg.train_checkpoint_black, map_location=cfg.device)
        agent_black.load_state_dict(state_dict)

    if cfg.get("train_checkpoint_white"):
        log.info(f"Loading white agent checkpoint from {cfg.train_checkpoint_white}")
        state_dict = torch.load(cfg.train_checkpoint_white, map_location=cfg.device)
        agent_white.load_state_dict(state_dict)

    # 3) VersusPlayCollector 설정
    log.info("--- 4. Setting up VersusPlay Collector ---")
    collector = VersusPlayCollector(
        env,
        policy_black=agent_black,
        policy_white=agent_white,
    )

    # 4) 결과 저장 디렉토리
    log.info(f"Results will be saved to: {run_dir}")
    os.makedirs(run_dir, exist_ok=True)

    # 5) 학습 루프
    log.info(f"--- 5. Starting Training ({cfg.epochs} epochs, VersusPlay) ---")
    for epoch in range(cfg.epochs):
        log.info(f"\n[Epoch {epoch + 1}/{cfg.epochs}] Collecting data...")

        # VersusPlayCollector는 (blacks, whites, info)를 반환함
        transitions_black, transitions_white, info = collector.rollout(cfg.steps)

        # 5-1) 흑 학습
        if transitions_black is not None and len(transitions_black) > 0:
            log.info(f"[Epoch {epoch + 1}] Learning BLACK from data...")
            learn_info_black = agent_black.learn(transitions_black) or {}
        else:
            learn_info_black = {}
            log.warning(
                f"[Epoch {epoch + 1}] No black data collected. Skipping black learning."
            )

        # 5-2) 백 학습
        if transitions_white is not None and len(transitions_white) > 0:
            log.info(f"[Epoch {epoch + 1}] Learning WHITE from data...")
            learn_info_white = agent_white.learn(transitions_white) or {}
        else:
            learn_info_white = {}
            log.warning(
                f"[Epoch {epoch + 1}] No white data collected. Skipping white learning."
            )

        # 5-3) 로그 정보 합치기 (키에 prefix 붙여서 충돌 방지)
        for k, v in learn_info_black.items():
            info[f"black_{k}"] = v
        for k, v in learn_info_white.items():
            info[f"white_{k}"] = v

        # 5-4) 에포크 로그 출력
        log.info(f"--- Epoch:{epoch:03d} Results ---")
        log.info(f"FPS (Data Collection): {info.get('fps', 0):.2f}")
        log.info(f"Black Win Rate: {info.get('black_win', 0):.2%}")
        log.info(f"White Win Rate: {info.get('white_win', 0):.2%}")
        log.info(f"Black Total Loss: {info.get('black_total_loss', 'N/A')}")
        log.info(f"White Total Loss: {info.get('white_total_loss', 'N/A')}")
        log.info("------------------------------")

        # 5-5) 체크포인트 저장 (예: 10 에포크마다)
        if (epoch + 1) % 10 == 0:
            save_path_black = os.path.join(
                run_dir, f"VersusPlay_black_epoch_{epoch + 1}.pt"
            )
            save_path_white = os.path.join(
                run_dir, f"VersusPlay_white_epoch_{epoch + 1}.pt"
            )
            torch.save(agent_black.state_dict(), save_path_black)
            torch.save(agent_white.state_dict(), save_path_white)
            log.info(
                f"Saved checkpoints to {save_path_black} and {save_path_white}"
            )

        # 여기서는 _refresh_collector_models() 같은 갱신/리셋 함수 사용 안 함.
        # collector는 agent_black / agent_white 객체를 참조하고 있으므로,
        # learn() 호출로 업데이트된 파라미터가 다음 rollout에 그대로 반영됨.
        # 에포크마다 환경을 완전히 리셋하고 싶다면 아래 한 줄만 추가 가능:
        # collector.reset()

    # 최종 모델 저장
    final_black = os.path.join(run_dir, "VersusPlay_black_final.pt")
    final_white = os.path.join(run_dir, "VersusPlay_white_final.pt")
    torch.save(agent_black.state_dict(), final_black)
    torch.save(agent_white.state_dict(), final_white)
    log.info(f"Final models saved to {final_black} and {final_white}")


# --- Hydra 엔트리포인트 ---
@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
def main(cfg: DictConfig):

    # --- 0. run_dir 및 로거 설정 ---
    # Colab Google Drive 경로 또는 로컬 경로
    base_run_dir = cfg.get("run_dir", ".")
    os.makedirs(base_run_dir, exist_ok=True)  # 상위 저장 폴더 생성

    # base_run_dir/timestamp 형태의 하위 폴더 생성
    run_dir = os.path.join(base_run_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # 이 시점부터 모든 log.info가 콘솔 + run_dir/train.log에 기록됨
    setup_logger(run_dir)

    log.info("--- 1. Configuration ---")
    log.info(OmegaConf.to_yaml(cfg))
    log.info(f"Results will be saved to: {run_dir}")

    # 시드 고정
    set_seed(cfg.seed)

    # --- 1. 환경(Env) 생성 ---
    log.info(f"--- 2. Creating Environment ({cfg.num_envs} parallel envs) ---")
    env = GomokuEnv(
        num_envs=cfg.num_envs,
        board_size=cfg.board_size,
        device=cfg.device,
    )

    # collector_type이 VersusPlay이면 VersusPlay 전용 루프로 진입
    if cfg.collector_type == "VersusPlay":
        log.info("--- 3. VersusPlay Mode selected ---")
        run_versus_play_training(cfg, env, run_dir)
        log.info("--- Training Finished (VersusPlay) ---")
        return

    # --- 2. 에이전트(Agent) 생성 (BlackPlay / WhitePlay / SelfPlay용) ---
    log.info(f"--- 3. Creating Agent ({cfg.algo.name}) ---")
    agent = get_policy(
        name=cfg.algo.name,
        cfg=cfg.algo,
        action_spec=env.action_spec,
        observation_spec=env.observation_spec,
        device=cfg.device,
    )

    # 이미 학습된 가중치로부터 이어서 학습하는 경우
    if cfg.get("train_checkpoint"):
        log.info(f"Loading training agent checkpoint from {cfg.train_checkpoint}")
        state_dict = torch.load(cfg.train_checkpoint, map_location=cfg.device)
        agent.load_state_dict(state_dict)
        log.info("Training agent checkpoint loaded.")

    # --- 3. 데이터 수집기(Collector) 설정 ---
    log.info(f"--- 4. Setting up {cfg.collector_type} Collector ---")
    opponent_policy = uniform_policy

    opponent_cfg = cfg.get("opponent", {})
    opponent_type = opponent_cfg.get("type", "random")

    # 이미 학습된 모델을 상대(agent)로 사용하는 경우
    if opponent_type == "pretrained":
        checkpoint_path = opponent_cfg.get("checkpoint")
        if not checkpoint_path:
            raise ValueError("Opponent type is 'pretrained' but no checkpoint was provided.")
        opponent_algo_name = opponent_cfg.get("algo_name", cfg.algo.name)
        opponent_algo_cfg = opponent_cfg.get("algo_cfg", cfg.algo)
        log.info(
            "Loading opponent checkpoint from %s using algo %s",
            checkpoint_path,
            opponent_algo_name,
        )
        opponent_policy = get_pretrained_policy(
            name=opponent_algo_name,
            cfg=opponent_algo_cfg,
            action_spec=env.action_spec,
            observation_spec=env.observation_spec,
            checkpoint_path=checkpoint_path,
            device=cfg.device,
        )
        opponent_policy.eval()
        log.info("Opponent model loaded.")
    elif opponent_type != "random":
        raise ValueError(f"Unknown opponent type: {opponent_type}")

    if cfg.collector_type == "BlackPlay":
        collector = BlackPlayCollector(
            env,
            policy_black=agent,          # 학습할 대상
            policy_white=opponent_policy # 고정된 상대
        )
    elif cfg.collector_type == "WhitePlay":
        collector = WhitePlayCollector(
            env,
            policy_black=opponent_policy, # 고정된 상대
            policy_white=agent            # 학습할 대상
        )
    elif cfg.collector_type == "SelfPlay":
        collector = SelfPlayCollector(env, agent)
    else:
        raise ValueError(f"Unknown collector_type: {cfg.collector_type}")

    # --- 4. 학습 루프 시작 ---
    log.info(f"--- 5. Starting Training ({cfg.epochs} epochs) ---")

    for epoch in range(cfg.epochs):
        log.info(f"\n[Epoch {epoch + 1}/{cfg.epochs}] Collecting data...")
        # 1. 데이터 수집
        transitions, info = collector.rollout(cfg.steps)

        # 2. 에이전트 학습
        if transitions is not None and len(transitions) > 0:
            log.info(f"[Epoch {epoch + 1}/{cfg.epochs}] Learning from data...")
            info.update(agent.learn(transitions))
        else:
            log.warning(
                f"[Epoch {epoch + 1}/{cfg.epochs}] No data collected. Skipping learning."
            )

        # 3. 로그 출력
        log.info(f"--- Epoch:{epoch:03d} Results ---")
        log.info(f"FPS (Data Collection): {info.get('fps', 0):.2f}")
        log.info(f"Black Win Rate: {info.get('black_win', 0):.2%}")
        log.info(f"White Win Rate: {info.get('white_win', 0):.2%}")
        log.info(
            "Losses - raw_loss: %s, policy: %s, value: %s, entropy: %s",
            info.get("loss", "N/A"),
            info.get("loss_objective", "N/A"),
            info.get("loss_critic", "N/A"),
            info.get("loss_entropy", "N/A"),
        )
        log.info(
            "Grad/Adv - grad_norm: %s, adv_mean: %s, adv_std: %s",
            info.get("grad_norm", "N/A"),
            info.get("advantage_meam", "N/A"),  # learn 쪽 오타 그대로 사용
            info.get("advantage_std", "N/A"),
        )
        log.info("------------------------------")

        # (예: 5 에포크마다 모델 저장)
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(
                run_dir, f"{cfg.collector_type}_agent_epoch_{epoch + 1}.pt"
            )
            torch.save(agent.state_dict(), save_path)
            log.info(f"Model checkpoint saved to {save_path}")

    log.info("--- Training Finished ---")

    # 최종 모델 저장
    final_save_path = os.path.join(run_dir, f"{cfg.collector_type}_agent_final.pt")
    torch.save(agent.state_dict(), final_save_path)
    log.info(f"Final model saved to {final_save_path}")


if __name__ == "__main__":
    main()
