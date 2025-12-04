import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import torch
import time
import random
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QAction,
    QFileDialog,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QKeySequence
import logging
from tensordict import TensorDict
from torchrl.data.tensor_specs import (
    DiscreteTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    BinaryDiscreteTensorSpec,
)

# --- 기존 demo.py에서 GomokuBoard 클래스를 임포트합니다 ---
# (demo.py가 scripts/demo.py에 있다고 가정)
try:
    from scripts.demo import GomokuBoard, Piece
except ImportError:
    print("=" * 50)
    print("ERROR: demo_model_duel.py는 scripts/demo.py 파일이 필요합니다.")
    print("scripts/demo.py에서 GomokuBoard 클래스를 임포트할 수 없습니다.")
    print("=" * 50)
    sys.exit(1)

# --- 정책 로드 관련 임포트 ---
from src.policy import get_policy
from src.policy.base import Policy

"""
이 스크립트는 demo.py의 GomokuBoard GUI와
model_duel.py의 AI-vs-AI 로직을 결합합니다.
"""


# --- model_duel.py에서 가져온 헬퍼 함수 ---
def _build_specs(cfg: DictConfig):
    board_size = cfg.board_size
    device = cfg.device
    num_envs = cfg.num_envs  # 1

    action_spec = DiscreteTensorSpec(
        board_size * board_size,
        shape=[
            num_envs,
        ],
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
        shape=[
            num_envs,
        ],
        device=device,
    )

    return action_spec, observation_spec


# --- PPO vs A2C를 위해 algo_cfg를 받는 수정된 로더 ---
def _load_policy(
    checkpoint_path: str,
    algo_cfg: DictConfig,
    action_spec,
    observation_spec,
    device,
):
    """
    특정 알고리즘 설정(algo_cfg)을 사용하여 정책을 로드합니다.
    """
    policy = get_policy(
        name=algo_cfg.name,  # ppo 또는 a2c
        cfg=algo_cfg,
        action_spec=action_spec,
        observation_spec=observation_spec,
        device=device,
    )
    # .pt 파일 로드
    state_dict = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()  # 추론 모드로 설정
    return policy


@hydra.main(version_base=None, config_path="../configs", config_name="gui_duel_config")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    app = QApplication(sys.argv)

    # 1. 스펙 생성
    action_spec, observation_spec = _build_specs(cfg)

    # 2. 흑/백 정책 로드 (PPO vs A2C 가능)
    logging.info(f"Loading Black policy ({cfg.algo.name})...")
    black_policy = _load_policy(
        cfg.black_checkpoint,
        cfg.algo,
        action_spec,
        observation_spec,
        cfg.device,
    )

    logging.info(f"Loading White policy ({cfg.algo.name})...")
    white_policy = _load_policy(
        cfg.white_checkpoint,
        cfg.algo,
        action_spec,
        observation_spec,
        cfg.device,
    )

    policies = {Piece.BLACK: black_policy, Piece.WHITE: white_policy}

    # 3. GUI 보드 생성 (중요: human_color=None으로 설정)
    board = GomokuBoard(
        grid_size=cfg.get("grid_size", 28),
        piece_radius=cfg.get("piece_radius", 12),
        board_size=cfg.get("board_size", 15),
        human_color=None,  # <-- 사람이 플레이하지 않음
        model=None,  # <-- 사용되지 않음
    )

    # 4. 윈도우 및 메뉴 설정 (demo.py와 거의 동일)
    window = QMainWindow()
    window.setMinimumSize(board.sizeHint())
    window.setWindowTitle(
        f"Gomoku AI Duel: {cfg.algo.name}(B) vs {cfg.algo.name}(W)"
    )

    central_widget = QWidget()
    label = QLabel()
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setFont(QFont("Arial", 20))
    label.setText("Starting Duel...")
    label.setMinimumHeight(32)
    layout = QVBoxLayout(central_widget)
    layout.addWidget(label, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignVCenter)
    layout.addWidget(board, Qt.AlignmentFlag.AlignCenter)
    window.setCentralWidget(central_widget)

    menu = window.menuBar().addMenu("&Menu")
    status_bar = window.statusBar()
    
    # (참고: Open 메뉴는 이 스크립트에선 큰 의미 없음)
    open_action = QAction("&Open (Not Recommended)")
    menu.addAction(open_action)

    reset_action = QAction("&Reset")
    reset_action.triggered.connect(board.reset)
    reset_action.setShortcut("Ctrl+R")
    reset_action.setToolTip("Reset the board.")
    menu.addAction(reset_action)

    # 5. AI 대결을 위한 게임 루프 (QTimer 사용)

    def game_step():
        """
        AI-vs-AI의 한 스텝을 진행합니다.
        """
        # 1. 현재 보드가 비어있는지(첫 수인지) 확인
        is_first_move = (board.latest_move is None)

        # 2. 게임 종료 체크
        if board.done:
            if board.current_player == Piece.BLACK:
                winner = "WHITE"
            else:
                winner = "BLACK"
            logging.info(f"Game over! Winner: {winner}")
            label.setText(f"Game Over! Winner: {winner}")
            status_bar.showMessage(f"Game Over! Winner: {winner}", 5000)
            return
        current_player_piece = board.current_player
        # 3. 랜덤 오프닝 로직 적용
        if is_first_move:
            label.setText(f"Random Opening (5x5 Center)")
            QApplication.processEvents()
            # 중앙 좌표 및 범위 설정 (5x5)
            center = board.board_size // 2
            margin = 2 # center-2 ~ center+2 (총 5칸)
            # 유효한 수(이미 돌이 없는 곳)를 찾을 때까지 반복 (첫 수라 사실 항상 비어있음)
            while True:
                x = random.randint(center - margin, center + margin)
                y = random.randint(center - margin, center + margin)
                if board._is_action_valid(x * board.board_size + y):
                    break

            logging.info(f"Random Start | {current_player_piece.name} -> ({x}, {y})")
            board.step([x, y])

            # 다음 턴 예약
            QTimer.singleShot(int(cfg.sleep_per_move * 1000), game_step)
            return

        # --- 5-2. 현재 턴의 AI 정책 선택 ---
        policy = policies[current_player_piece]
        label.setText(f"Thinking: {current_player_piece.name}")
        QApplication.processEvents() # 라벨 업데이트

        # --- 5-3. 텐서 생성 ---
        tensordict = TensorDict(
            {
                "observation": board._env.get_encoded_board(),
                "action_mask": board._env.get_action_mask(),
            },
            batch_size=[cfg.num_envs], # 1
            device=cfg.device,
        )

        # --- 5-4. AI 추론 ---
        with torch.no_grad():
            tensordict = policy(tensordict).cpu()
        action: int = tensordict["action"].item()
        x, y = divmod(action, board.board_size)

        # --- 5-5. GUI 및 환경 업데이트 ---
        logging.info(f"Move | {current_player_piece.name} -> ({x}, {y})")
        board.step([x, y]) # (내부적으로 board.update() 호출됨)
        QApplication.processEvents() # 보드 새로 그리기

        # --- 5-6. 다음 턴 예약 ---
        QTimer.singleShot(int(cfg.sleep_per_move * 1000), game_step)

    # 6. 게임 시작
    window.show()
    # 0.5초 후 첫 번째 수를 두도록 예약
    QTimer.singleShot(500, game_step)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
