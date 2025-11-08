import torch
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from src.envs import GomokuEnv 
from src.utils.policy import uniform_policy 
from src.utils.misc import set_seed
from src.collectors.collectors import BlackPlayCollector, WhitePlayCollector, SelfPlayCollector
from src.policy import get_policy

# 로거 설정
log = logging.getLogger(__name__)

# --- 2. Hydra: configs/ 폴더의 .yaml 파일을 읽어옴 ---
@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
def main(cfg: DictConfig):
    
    # --- 0. 설정 파일 내용 출력 ---
    log.info("--- 1. Configuration ---")
    log.info(OmegaConf.to_yaml(cfg))
    
    # 시드 고정
    set_seed(cfg.seed)
    
    # --- 1. 환경(Env) 생성 ---
    log.info(f"--- 2. Creating Environment ({cfg.num_envs} parallel envs) ---")
    env = GomokuEnv(
        num_envs=cfg.num_envs, 
        board_size=cfg.board_size, 
        device=cfg.device
    )

    # --- 2. 에이전트(Agent) 생성 ---
    log.info(f"--- 3. Creating Agent ({cfg.algo.name}) ---")
    # (src.policy.get_policy 함수가 PPO/A2C를 반환)
    agent = get_policy(
        name=cfg.algo.name,
        cfg=cfg.algo, 
        action_spec=env.action_spec, 
        observation_spec=env.observation_spec, 
        device=cfg.device
    )

    # --- 3. 데이터 수집기(Collector) 설정 ---
    log.info(f"--- 4. Setting up {cfg.collector_type} Collector ---")
    random_opponent = uniform_policy
    
    if cfg.collector_type == "BlackPlay":
        collector = BlackPlayCollector(
            env, 
            policy_black=agent,           # 학습할 대상
            policy_white=random_opponent  # 고정된 상대
        )
    elif cfg.collector_type == "WhitePlay":
        collector = WhitePlayCollector(
            env,
            policy_black=random_opponent, # 고정된 상대
            policy_white=agent            # 학습할 대상
        )
    elif cfg.collector_type == "SelfPlay":
        collector = SelfPlayCollector(env, agent)
    else:
        raise ValueError(f"Unknown collector_type: {cfg.collector_type}")

    # --- 4. 학습 루프 시작 ---
    log.info(f"--- 5. Starting Training ({cfg.epochs} epochs) ---")
    
    # Colab Google Drive 경로 또는 로컬 경로
    run_dir = cfg.get("run_dir", ".") 
    log.info(f"Results will be saved to: {run_dir}")
    os.makedirs(run_dir, exist_ok=True) # 저장 폴더 생성

    for epoch in range(cfg.epochs):
        
        log.info(f"\n[Epoch {epoch+1}/{cfg.epochs}] Collecting data...")
        # 1. 데이터 수집
        # (cfg.collector_type에 따라 반환값이 달라질 수 있으므로, 
        #  BlackPlayCollector 기준으로 우선 작성)
        transitions, info = collector.rollout(cfg.steps)
        
        # 2. 에이전트 학습
        if transitions is not None and len(transitions) > 0:
            log.info(f"[Epoch {epoch+1}/{cfg.epochs}] Learning from data...")
            info.update(agent.learn(transitions))
        else:
            log.warning(f"[Epoch {epoch+1}/{cfg.epochs}] No data collected. Skipping learning.")
        
        # 3. 로그 출력
        log.info(f"--- Epoch:{epoch:03d} Results ---")
        log.info(f"FPS (Data Collection): {info.get('fps', 0):.2f}")
        log.info(f"Black Win Rate: {info.get('black_win', 0):.2%}")
        log.info(f"White Win Rate: {info.get('white_win', 0):.2%}")
        log.info(f"Total Loss: {info.get('total_loss', 'N/A')}")
        log.info("------------------------------")
        
        # (참고: 모델 저장 로직 - 예: 10 에포크마다 저장)
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(run_dir, f"{cfg.collector_type}_agent_epoch_{epoch+1}.pt")
            # torch.save(agent.state_dict(), save_path)
            # log.info(f"Model checkpoint saved to {save_path}")

    log.info("--- Training Finished ---")
    
    # 최종 모델 저장
    final_save_path = os.path.join(run_dir, f"{cfg.collector_type}_agent_final.pt")
    # torch.save(agent.state_dict(), final_save_path)
    # log.info(f"Final model saved to {final_save_path}")


if __name__ == "__main__":
    main()