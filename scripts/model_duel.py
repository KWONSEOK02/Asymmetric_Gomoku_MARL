import logging
import time

import hydra
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torchrl.data.tensor_specs import (
    BinaryDiscreteTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)

from src.envs.core import Gomoku
from src.policy import get_policy


def _build_specs(cfg: DictConfig):
    board_size = cfg.board_size
    device = cfg.device
    num_envs = cfg.num_envs

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


def _load_policy(checkpoint_path: str, cfg: DictConfig, action_spec, observation_spec):
    policy = get_policy(
        name=cfg.algo.name,
        cfg=cfg.algo,
        action_spec=action_spec,
        observation_spec=observation_spec,
        device=cfg.device,
    )
    state_dict = torch.load(checkpoint_path, map_location=cfg.device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def _format_board(board: torch.Tensor) -> str:
    symbols = {0: ".", 1: "B", -1: "W"}
    rows = []
    for row in board[0]:
        rows.append(" ".join(symbols[int(cell.item())] for cell in row))
    return "\n".join(rows)


@hydra.main(version_base=None, config_path="../configs", config_name="duel")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)

    action_spec, observation_spec = _build_specs(cfg)
    black_policy = _load_policy(cfg.black_checkpoint, cfg, action_spec, observation_spec)
    white_policy = _load_policy(cfg.white_checkpoint, cfg, action_spec, observation_spec)

    env = Gomoku(num_envs=cfg.num_envs, board_size=cfg.board_size, device=cfg.device)
    env.reset()

    logging.info("Starting duel: Black (%s) vs White (%s)", cfg.black_checkpoint, cfg.white_checkpoint)
    move_number = 1

    while not env.done.item():
        current_player = "Black" if env.turn.item() == 0 else "White"
        policy = black_policy if env.turn.item() == 0 else white_policy

        tensordict = TensorDict(
            {
                "observation": env.get_encoded_board(),
                "action_mask": env.get_action_mask(),
            },
            batch_size=[cfg.num_envs],
            device=cfg.device,
        )

        with torch.no_grad():
            tensordict = policy(tensordict)

        action = tensordict["action"].view(-1)[0].item()
        row, col = divmod(action, cfg.board_size)

        if not env.is_valid(torch.tensor([action], device=cfg.device)).item():
            raise ValueError(f"{current_player} selected an invalid move: ({row}, {col})")

        env.step(torch.tensor([action], device=cfg.device))

        logging.info("Move %02d | %s -> (%d, %d)", move_number, current_player, row, col)
        print(_format_board(env.board))
        print()

        move_number += 1
        if env.done.item():
            break

        time.sleep(cfg.sleep_per_move)

    winner = "Black" if env.turn.item() == 1 else "White"
    logging.info("Game over! Winner: %s", winner)


if __name__ == "__main__":
    main()
