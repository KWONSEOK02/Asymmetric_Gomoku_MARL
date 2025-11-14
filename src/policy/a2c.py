from typing import Dict

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torchrl.data import TensorSpec, DiscreteTensorSpec
from torch.cuda import _device_t
from omegaconf import DictConfig
from torchrl.objectives.value.functional import (
    vec_generalized_advantage_estimate,
)

from .base import Policy
from .common import (
    make_ppo_ac,
    get_optimizer,
    make_critic,
    make_ppo_actor,
)

from src.utils.module import count_parameters  # optional, 안 쓰면 삭제해도 됨


class A2C(Policy):
    def __init__(
        self,
        cfg: DictConfig,
        action_spec: DiscreteTensorSpec,
        observation_spec: TensorSpec,
        device: _device_t = "cuda",
    ) -> None:
        super().__init__(cfg, action_spec, observation_spec, device)

        self.cfg: DictConfig = cfg
        self.device: _device_t = device

        # 하이퍼파라미터
        self.entropy_coef: float = cfg.entropy_coef
        self.gae_gamma: float = cfg.gamma
        self.gae_lambda: float = cfg.gae_lambda
        self.average_gae: bool = cfg.get("average_gae", False)

        self.max_grad_norm: float = cfg.max_grad_norm
        self.normalize_advantage: bool = cfg.get("normalize_advantage", True)

        # critic loss 계수 (없으면 0.5)
        self.value_loss_coef: float = cfg.get("value_loss_coef", 0.5)

        # gradient accumulation 용 chunk size (transition 개수 기준)
        # 전체 배치를 한 번에 쓰되, backward만 chunk 단위로 쪼갬
        self.chunk_size: int = int(cfg.get("chunk_size", 8192))

        # Actor / Critic 구성
        if self.cfg.get("share_network"):
            actor_value_operator = make_ppo_ac(
                cfg, action_spec=action_spec, device=self.device
            )
            self.actor = actor_value_operator.get_policy_operator()
            self.critic = actor_value_operator.get_value_head()
        else:
            self.actor = make_ppo_actor(
                cfg=cfg, action_spec=action_spec, device=self.device
            )
            self.critic = make_critic(cfg=cfg, device=self.device)

        # lazy module 워밍업
        fake_input = observation_spec.zero()
        fake_input["action_mask"] = ~fake_input["action_mask"]
        with torch.no_grad():
            self.actor(fake_input)
            self.critic(fake_input)
        # print(f"actor params: {count_parameters(self.actor)}")
        # print(f"critic params: {count_parameters(self.critic)}")

        # actor + critic 같이 업데이트
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optim = get_optimizer(self.cfg.optimizer, params)

    # ------------------------------------------------------------------
    # 환경 상호작용
    # ------------------------------------------------------------------
    def __call__(self, tensordict: TensorDict) -> TensorDict:
        tensordict = tensordict.to(self.device)

        # Actor
        actor_input = tensordict.select("observation", "action_mask", strict=False)
        actor_output: TensorDict = self.actor(actor_input)
        actor_output = actor_output.exclude("probs")  # rollout에는 probs 필요 X
        tensordict.update(actor_output)

        # Critic
        critic_input = tensordict.select("hidden", "observation", strict=False)
        critic_output = self.critic(critic_input)
        tensordict.update(critic_output)

        return tensordict

    # ------------------------------------------------------------------
    # 학습: gradient accumulation + full batch 1-step
    # ------------------------------------------------------------------
    def learn(self, data: TensorDict) -> Dict:
        """
        data batch shape: [num_envs, T]

        keys (PPO와 동일):
          - "state_value"
          - ("next", "state_value")
          - ("next", "reward")
          - ("next", "done")
          - "observation", "action_mask", "action", ...
        """

        # ----- 1. GAE 계산 -----
        value = data["state_value"].to(self.device)
        next_value = data["next", "state_value"].to(self.device)
        done = data["next", "done"].unsqueeze(-1).to(self.device)
        reward = data["next", "reward"].to(self.device)

        with torch.no_grad():
            adv, value_target = vec_generalized_advantage_estimate(
                self.gae_gamma,
                self.gae_lambda,
                value,
                next_value,
                reward,
                done=done,
                terminated=done,
                time_dim=data.ndim - 1,  # 마지막 축이 time
            )

            loc = adv.mean()
            scale = adv.std().clamp_min(1e-4)

            if self.average_gae:
                adv = (adv - loc) / scale

            data.set("advantage", adv)
            data.set("value_target", value_target)

        # ----- 2. invalid transition 제거 -----
        invalid = data.get("invalid", None)
        if invalid is not None:
            data = data[~invalid]

        # [num_envs, T] -> [B]
        data = data.reshape(-1)
        batch_size = data.batch_size[0]

        # guard: 데이터가 없으면 noop
        if batch_size == 0:
            return {
                "advantage_meam": loc.item(),
                "advantage_std": scale.item(),
                "grad_norm": 0.0,
                "loss": 0.0,
                "loss_objective": 0.0,
                "loss_critic": 0.0,
                "loss_entropy": 0.0,
            }

        self.train()
        self.optim.zero_grad()

        # 로깅용 누적 값 (full batch 평균이 되도록 weight 적용)
        total_loss_acc = 0.0
        policy_loss_acc = 0.0
        value_loss_acc = 0.0
        entropy_loss_acc = 0.0

        # ----- 3. chunk 단위 backward (gradient accumulation) -----
        for start in range(0, batch_size, self.chunk_size):
            end = min(start + self.chunk_size, batch_size)
            mini: TensorDict = data[start:end].to(self.device)
            chunk_bsz = end - start
            weight = float(chunk_bsz) / float(batch_size)  # full batch mean 맞추기

            advantage = mini["advantage"].squeeze(-1)
            value_target = mini["value_target"].squeeze(-1)

            # --- 3-1. Actor forward: 여기서 hidden도 생성됨 ---
            actor_input = mini.select("observation", "action_mask", strict=False)
            actor_output: TensorDict = self.actor(actor_input)

            probs = actor_output.get("probs", None)
            if probs is None:
                raise KeyError(
                    "A2C actor output에 'probs' 키가 없습니다. "
                    "Categorical 분포를 만들 수 없어 log_prob를 계산할 수 없습니다."
                )

            action = mini["action"].squeeze(-1)
            dist = torch.distributions.Categorical(probs=probs)
            log_prob = dist.log_prob(action)

            # Policy loss: -E[ A * log π(a|s) ]
            policy_loss = -(log_prob * advantage).mean()

            # --- 3-2. Critic forward: **actor_output에서 hidden 재사용** ---
            critic_input = actor_output.select("hidden", "observation", strict=False)
            critic_output = self.critic(critic_input)
            value_pred = critic_output["state_value"].squeeze(-1)

            value_loss = F.smooth_l1_loss(value_pred, value_target)
            value_loss = self.value_loss_coef * value_loss

            # --- 3-3. Entropy bonus ---
            entropy = dist.entropy().mean()
            entropy_loss = -self.entropy_coef * entropy

            loss = policy_loss + value_loss + entropy_loss

            # full batch 평균 gradient와 동일해지도록 weight 곱
            (loss * weight).backward()

            # 로그 값도 동일한 방식으로 가중 평균
            total_loss_acc += loss.item() * weight
            policy_loss_acc += policy_loss.item() * weight
            value_loss_acc += value_loss.item() * weight
            entropy_loss_acc += entropy_loss.item() * weight

        # ----- 4. grad clipping + optimizer.step() 딱 한 번 -----
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            self.max_grad_norm,
        )
        self.optim.step()

        self.eval()

        return {
            # PPO 쪽 로깅 키랑 맞추려고 오타 그대로 유지
            "advantage_meam": loc.item(),
            "advantage_std": scale.item(),
            "grad_norm": grad_norm.item(),
            "loss": total_loss_acc,
            "loss_objective": policy_loss_acc,
            "loss_critic": value_loss_acc,
            "loss_entropy": entropy_loss_acc,
        }

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict):
        self.critic.load_state_dict(state_dict["critic"], strict=False)
        self.actor.load_state_dict(state_dict["actor"])

        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optim = get_optimizer(self.cfg.optimizer, params)

    # ------------------------------------------------------------------
    # train / eval 모드
    # ------------------------------------------------------------------
    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()