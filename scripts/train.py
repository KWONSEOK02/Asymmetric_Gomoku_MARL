import torch
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

# --- 1. src/ 폴더의 우리 코드를 import ---
# (이 import가 작동하려면 src/agents/에 PPO/A2C 클래스가 구현되어 있어야 합니다)
from src.envs.gomoku_env import GomokuEnv
from src.utils.policy import uniform_policy 
from src.utils.misc import set_seed
from src.agents.collectors import BlackPlayCollector, WhitePlayCollector, SelfPlayCollector
from src.agents import get_policy  # (이 함수는 src/agents/__init__.py에 구현 필요)

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
    # (get_policy 함수가 PPO/A2C를 반환한다고 가정)
    agent = get_policy(
        name=cfg.algo.name,
        cfg=cfg.algo, 
        action_spec=env.action_spec, 
        observation_spec=env.observation_spec, 
        device=cfg.device
    )

    # --- 3. 데이터 수집기(Collector) 설정 ---
    # (예시: '흑' 에이전트를 '무작위 봇' 상대로 학습)
    log.info(f"--- 4. Setting up BlackPlayCollector ---")
    random_opponent = uniform_policy
    collector = BlackPlayCollector(
        env, 
        policy_black=agent,           # 학습할 대상
        policy_white=random_opponent  # 고정된 상대 (또는 pretrained 모델)
    )
    
    # (참고: cfg.collector_type == "White"일 때 WhitePlayCollector를 부르는 로직 추가 필요)

    # --- 4. 학습 루프 시작 ---
    log.info(f"--- 5. Starting Training ({cfg.epochs} epochs) ---")
    
    output_dir = hydra.utils.get_original_cwd() # Hydra가 생성한 results 폴더
    run_dir = cfg.get("run_dir", output_dir) # Colab Google Drive 경로
    log.info(f"Results will be saved to: {run_dir}")

    for epoch in range(cfg.epochs):
        
        log.info(f"\n[Epoch {epoch+1}/{cfg.epochs}] Collecting data...")
        # 1. 데이터 수집
        # (CPU에서는 이 rollout 단계가 매우 오래 걸릴 수 있음)
        black_transitions, info = collector.rollout(cfg.steps_per_epoch)
        
        # 2. 에이전트 학습
        if black_transitions is not None and len(black_transitions) > 0:
            log.info(f"[Epoch {epoch+1}/{cfg.epochs}] Learning from data...")
            info.update(agent.learn(black_transitions))
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
            save_path = os.path.join(run_dir, f"black_agent_epoch_{epoch+1}.pt")
            # torch.save(agent.state_dict(), save_path)
            # log.info(f"Model checkpoint saved to {save_path}")

    log.info("--- Training Finished ---")
    
    # 최종 모델 저장
    final_save_path = os.path.join(run_dir, "black_final.pt")
    # torch.save(agent.state_dict(), final_save_path)
    # log.info(f"Final model saved to {final_save_path}")

if __name__ == "__main__":
    main()