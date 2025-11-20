# Asymmetric_Gomoku_MARL (오목 비대칭 멀티에이전트)

## 1. 📝 프로젝트 개요 (Project Overview)

본 프로젝트는 오목(Gomoku) 환경에서 멀티에이전트 강화학습(MARL)을 적용하는 것을 목표로 합니다. 오목은 흑(선공)이 압도적으로 유리한 비대칭 게임입니다.

본 프로젝트는 `hesic73/gomoku_rl`을 기반으로, **흑(공격자) 에이전트**와 **백(방어자) 에이전트**를 독립적으로 학습시켜 , **PPO와 A2C 알고리즘의 학습 효율성과 전략적 차이를 비교 분석**합니다.

---

## 2. 🛠️ Tech Stack

* **Environment:** Conda (로컬 가상 환경), Google Colab (GPU 학습)
* **Local (Conda):** Python 3.11.5, PyTorch 2.1.0, TorchRL 0.2.1
* **Colab (GPU):** Python 3.12, PyTorch 2.2.2+cu118, TorchRL 0.5.0
* **Config & Demo:** Hydra-core, OmegaConf, PyQt5
* **Utilities:** NumPy (<2.0), WandB, Matplotlib

---

## 3. 🚀 프로젝트 실행 방법 (Getting Started)

팀원 모두 CPU 환경이므로, "개발은 로컬(CPU), 학습은 Colab(GPU)" 워크플로우를 따릅니다.

### 1. 저장소 복제 (Clone)

```bash
git clone https://github.com/KWONSEOK02/Asymmetric_Gomoku_MARL.git
cd Asymmetric_Gomoku_MARL
```


### 2. 로컬 가상 환경 설정 (Conda)

#### Python 3.11.5로 'gomoku_marl_env' 환경 생성
```bash
conda create -n gomoku_marl_env python=3.11.5
conda activate gomoku_marl_env
```

#### requirements.txt로 모든 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 3. Colab을 이용한 GPU 학습 (Training)
#### 1.Colab 노트북을 열고 런타임 유형을 T4 GPU로 변경합니다.

#### 2.Google Drive를 마운트(연결)합니다.
```bash
from google.colab import drive
drive.mount('/content/drive')
```


#### 3. 프로젝트를 클론하고 라이브러리를 설치
```bash
# Git 클론 경로가 꼬이지 않도록, Colab 기본 디렉토리(/content)로 이동합니다.
%cd /content

# GitHub에서 프로젝트를 복제합니다. (마크다운 없이 URL만 사용)
!git clone https://github.com/KWONSEOK02/Asymmetric_Gomoku_MARL.git

# 복제된 폴더 안으로 작업 디렉터리를 이동합니다.
%cd Asymmetric_Gomoku_MARL

# 프로젝트 폴더 안에서 requirements.txt 파일로 라이브러리를 설치합니다.
!pip install -r requirements.txt

# 2. (신규) PyTorch/TorchRL은 Colab의 CUDA 버전에 맞춰 별도 설치
# (Colab 기본 PyTorch/TorchVision을 삭제한 후, cu118 버전에 맞는 torch 2.2.2와 torchrl 0.5.0을 설치합니다.)
!pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.2.2+cu118
!pip install --no-deps torchrl==0.5.0 tensordict==0.5.0
```

#### 4. Google Drive에 결과를 저장하며 학습을 실행
##### 예제 1: PPO로 흑(Black) 에이전트 학습 (기본값)

흑(선공) 에이전트가 랜덤 상대를 이기도록 학습 (5 에포크 테스트 예시)
```bash
#`scripts/train.py`는 Hydra를 사용하며, 커맨드 라인에서 `key=value`를 통해 설정을 덮어쓸 수 있습니다.
#(기본 설정: collector_type: BlackPlay, algo: ppo)
!python -m scripts.train \
    device=cuda \
    wandb.mode=disabled \
    run_dir='/content/drive/My Drive/Gomoku_Results' \
    epochs=5
```

##### 예제 2: PPO로 백(White) 에이전트 학습

collector_type을 WhitePlay로 덮어써서, 백(후공) 에이전트가 랜덤 상대를 이기도록 학습
```bash
!python -m scripts.train \
    device=cuda \
    wandb.mode=disabled \
    run_dir='/content/drive/My Drive/Gomoku_Results' \
    epochs=5 \
    collector_type=WhitePlay
```

##### 예제 3: A2C로 흑(Black) 에이전트 학습

algo를 a2c로 덮어써서, PPO 대신 A2C 알고리즘으로 흑 에이전트를 학습
```bash
!python -m scripts.train \
    device=cuda \
    wandb.mode=disabled \
    run_dir='/content/drive/My Drive/Gomoku_Results' \
    epochs=5 \
    algo=a2c
```

##### 예제 4: A2C로 백(White) 에이전트 학습

collector_type과 algo를 모두 덮어써서, A2C 알고리즘으로 백 에이전트를 학습
```bash
!python -m scripts.train \
    device=cuda \
    wandb.mode=disabled \
    run_dir='/content/drive/My Drive/Gomoku_Results' \
    epochs=5 \
    collector_type=WhitePlay \
    algo=a2c
```

##### 예제 5: 이미 학습된 상대를 두고 추가 학습하기 (PPO/A2C 공통)

`train_checkpoint`로 학습을 이어갈 모델을 지정하고, `opponent.type=pretrained` 및 `opponent.checkpoint`로 상대를 고정할 수 있습니다. 아래 예시는 흑(학습 대상) PPO 에이전트를, 1차 학습된 백 PPO 모델을 상대로 추가 학습하는 명령입니다.

```bash
!python -m scripts.train \
    device=cuda \
    wandb.mode=disabled \
    run_dir='/content/drive/My Drive/Gomoku_Results' \
    collector_type=BlackPlay \
    epochs=5 \
    train_checkpoint=results/models/black_epoch500.pt \
    opponent.type=pretrained \
    opponent.checkpoint=results/models/white_epoch500.pt \
    opponent.algo_name=ppo
```

동일한 설정에서 `algo=a2c opponent.algo_name=a2c`로 바꾸면 A2C 모델끼리도 같은 방식으로 대전 학습을 진행할 수 있습니다.


### 4. 로컬에서 데모 실행 (Demo)
#### 1. Google Drive의 Gomoku_Results 폴더에서 학습된 모델(예: black_final.pt)을 로컬 results/models/ 폴더로 다운로드합니다.

#### 2. 로컬의 (gomoku_marl_env) 환경에서 demo.py를 실행합니다.

```bash
# (gomoku_marl_env)
python scripts/demo.py device=cpu checkpoint=results/models/black_final.pt
```

## 4. 알고리즘 (Algorithms)

본 프로젝트는 두 가지 핵심 강화학습 알고리즘인 PPO와 A2C를 지원합니다.

### PPO (`ppo.py`)

여러 에포크(epoch)와 미니배치(mini-batch)를 사용해 수집된 데이터를 재사용(Off-Policy)하는 안정적이고 효율적인 알고리즘입니다. `ClipPPOLoss`를 사용합니다.

### A2C (`a2c.py`)

`ppo.py`와 달리, 수집한 데이터를 재사용하지 않고 **"1 배치 = 1 업데이트"**를 수행하는 전통적인 A2C (Advantage Actor-Critic)의 **On-Policy** 방식을 따릅니다.

본 구현은 A2C의 안정적인 학습을 위해 다음과 같은 현대적인 기법들이 적용되어 있습니다.

* **GAE (Generalized Advantage Estimation):** 1-step TD, n-step TD 방식과 비교 실험한 결과(그래프 참고), 가장 성능이 우수했던 **GAE**를 사용해 Advantage를 계산합니다.
* **실전 안정화:** Gradient Clipping, `smooth_l1_loss`, Advantage 정규화 등 `ppo.py`와 유사한 실전 안정화 기법이 포함되어 있습니다.
* **메모리 최적화 (그래디언트 누적):** Colab GPU 환경의 메모리 한계(OOM) 내에서 대용량 배치를 학습하기 위해 **그래디언트 누적(Gradient Accumulation)**이 구현되었습니다.

📄 [A2C 상세 구현 및 실험 문서 (Notion) 작성자 : cxado-6919 ](https://www.notion.so/a2c-py-2aa67d3af68780a6a26cea9213549602?source=copy_link)

## 5. 📁 프로젝트 구조
```
Asymmetric_Gomoku_MARL/
├── configs/
│   ├── algo/
│   │   ├── ppo.yaml         # (가져옴) PPO 알고리즘 설정
│   │   └── a2c.yaml         # (신규) A2C 알고리즘 설정
│   ├── baseline/
│   │   ├── ppo.yaml         # (가져옴) Baseline PPO 설정
│   │   └── a2c.yaml         # (신규) Baseline A2C 설정
│   └── train_config.yaml  # (가져옴) 메인 학습 설정
├── notebooks/
├── results/
│   ├── models/
│   └── plots/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── demo.py
├── src/
│   ├── __init__.py
│   ├── collectors/
│   │   ├── __init__.py
│   │   └── collectors.py    # (가져옴) BlackPlayCollector 등
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── gomoku_env.py    # (가져옴) GomokuEnv 클래스
│   │   └── core.py          # (가져옴) Gomoku 핵심 로직
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py
│   ├── models/            # (신규) 신경망 "설계도" (.py)
│   │   ├── __init__.py
│   │   └── base_model.py     # (신규) PPO/A2C 공용 베이스 모델
│   ├── policy/            # (신규) "알고리즘" 자체
│   │   ├── __init__.py
│   │   ├── base.py          # (가져옴) Policy 추상 클래스
│   │   ├── common.py        # (가져옴) PPO/A2C 공통 헬퍼
│   │   ├── ppo.py           # (가져옴) PPO 알고리즘 클래스
│   │   └── a2c.py           # (신규 구현) A2C 알고리즘 클래스
│   └── utils/             # (공통 유틸리티)
│       ├── __init__.py
│       ├── augment.py
│       ├── log.py
│       ├── misc.py
│       └── policy.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 6. 🤝 협업 가이드라인 (Contribution Guidelines)

### Git Workflow
- `master` (Production): 최종 배포 브랜치
- `develop` (Staging): 개발 완료 코드를 병합하는 메인 브랜치
- `feat/*`, `fix/*`, `docs/*`: 기능별, 목적별 브랜치

### 작업 흐름
1. `develop` 브랜치에서 `feat/my-new-feature` 브랜치를 생성하여 작업을 시작합니다.
2. 기능 완료 후 **Pull Request(PR)** 를 생성합니다.
3. 1명 이상의 팀원에게 **Approve(리뷰 승인)** 를 받습니다.
4. Merge 전, `develop` 최신 변경 사항을 `pull` 하여 충돌을 최소화합니다.

### 프로젝트 규칙
- **PR은 작은 단위로.** 하나의 PR은 하나의 기능에만 집중합니다.
- 세부 작업은 체크리스트로 관리합니다.
- 작업 충돌을 방지하기 위해 회의 중 역할을 명확히 나눕니다.

### 개발 가이드라인
- 코딩 스타일: **PEP8 준수**, `flake8` 권장
- 함수·클래스 주석: **Google Style Docstring**
- 모든 팀원은 동일한 `requirements.txt` 환경에서 개발합니다.

### 커밋 및 브랜치 컨벤션
- 커밋 메시지 및 브랜치는 **Conventional Commits 규칙 준수**
  - 예시:  
    - `feat(agent): Add A2C agent logic`  
    - `fix(reward): Fix reward calculation bug`

📄 상세 컨벤션 문서 (Notion)  
https://www.notion.so/27167d3af687803ca8c1ec0a66bbeb59?source=copy_link

---

## 📝 커밋 컨벤션
**Conventional Commits** 규칙 준수

- `feat:` 새 기능
- `fix:` 버그 수정
- `docs:` 문서 변경
- `style:` 코드 포매팅, 세미콜론
- `refactor:` 코드 리팩토링
- `perf:` 성능 개선
- `test:` 테스트 관련
- `chore:` 빌드/배포/패키지
- `ci:` CI 설정
- `revert:` 이전 커밋 되돌리기

**메시지 형식**
```
feat(agent): add PPO multi-agent training
fix(reward): correct negative reward assignment
perf(train): reduce inference time by caching model
```
---

## 🌱 Git 브랜치 네이밍 컨벤션 (요약)

- **feature/** → 새로운 기능 / 알고리즘 / 환경 추가  
  예) `feature/ppo-agent`, `feature/gomoku-env-enhancement`

- **fix/** → 버그 수정  
  예) `fix/reward-calculation`, `fix/model-forward-error`

- **hotfix/** → 긴급 수정  
  예) `hotfix/training-crash-epoch-100`

- **refactor/** → 코드 구조 개선  
  예) `refactor/env-clean-structure`, `refactor/agent-base-class`

- **docs/** → 문서 작업  
  예) `docs/update-readme-install-guide`, `docs/add-evaluation-results`

---