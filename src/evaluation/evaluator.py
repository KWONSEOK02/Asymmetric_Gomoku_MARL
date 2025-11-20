import torch
from tensordict import TensorDict
from src.envs.gomoku_env import GomokuEnv


def evaluate_win_rate(
    black_policy,
    white_policy,
    board_size: int = 15,
    num_envs: int = 100,
    device: str = "cuda",
):
    """
    두 정책(흑/백)을 GomokuEnv에서 대결시켜:
      - 누적 보상 (cumulative_reward, 흑 기준)
      - 평균 보상 (average_reward)
      - 승률 (win_rate, %)
      - 흑/백 승 수, 무승부 수
    를 반환합니다.
    """
    black_policy.eval()
    white_policy.eval()

    # 1. 환경 생성 및 초기화
    env = GomokuEnv(num_envs=num_envs, board_size=board_size, device=device)
    td = env.reset()           # {"observation", "action_mask", "done", ...}
    done = td.get("done")      # [num_envs], 보통 전부 False

    first_step = True          # prev_action 방어용 플래그

    with torch.no_grad():
        while not done.all():
            # ─────────────────────────────
            # (1) 환경별 현재 턴 (0=흑, 1=백)
            # ─────────────────────────────
            turn = env.gomoku.turn  # shape: [num_envs]

            # ─────────────────────────────
            # (2) 두 정책 모두 돌린 뒤 턴에 맞는 action 선택
            # ─────────────────────────────
            policy_input = td.select("observation", "action_mask")

            td_black = black_policy(policy_input.clone())
            td_white = white_policy(policy_input.clone())

            actions_black = td_black["action"]    # [num_envs]
            actions_white = td_white["action"]    # [num_envs]

            # 각 env별 현재 턴에 맞게 액션 선택
            actions = torch.where(turn == 0, actions_black, actions_white)

            # ─────────────────────────────
            # (3) 이미 끝난 env(done=True)는
            #     이전 action을 유지 (2-step 이상부터)
            # ─────────────────────────────
            if not first_step and "action" in td.keys():
                prev_action = td["action"]
                # 끝난 env는 prev_action 그대로, 진행 중 env만 새 actions
                actions = torch.where(done, prev_action, actions)

            # 선택된 액션을 td에 세팅
            td.set("action", actions)

            # ─────────────────────────────
            # (4) 환경 진행
            #     (GomokuEnv는 done=True env를 내부에서 무시한다고 가정)
            # ─────────────────────────────
            td = env.step(td)
            done = td.get("done")
            first_step = False

    # ─────────────────────────────
    # 2. 승패 결과 집계 (마지막 td의 stats 사용)
    # ─────────────────────────────
    stats = td.get("stats", None)

    # 방어 코드: stats 없으면 바로 에러로 알려주기
    if stats is None:
        raise RuntimeError(
            "GomokuEnv의 step() 결과 TensorDict에 'stats' 키가 없습니다. "
            "env에서 stats['black_win'], stats['white_win']을 채우도록 구현되어 있는지 확인하세요."
        )

    black_wins = stats["black_win"].sum().item()
    white_wins = stats["white_win"].sum().item()
    draws = num_envs - black_wins - white_wins

    # 흑 기준: 이기면 +1점
    cumulative_reward_black = float(black_wins)
    average_reward_black = cumulative_reward_black / float(num_envs)

    return {
        "cumulative_reward": cumulative_reward_black,      # 예: 85.0
        "average_reward": average_reward_black,            # 예: 0.85
        "win_rate": f"{average_reward_black * 100:.1f}%",  # 예: "85.0%"
        "black_wins": int(black_wins),
        "white_wins": int(white_wins),
        "draws": int(draws),
    }

