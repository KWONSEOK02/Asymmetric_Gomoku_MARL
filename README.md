# Asymmetric_Gomoku_MARL (오목 비대칭 멀티에이전트)

## 1. 📝 프로젝트 개요 (Project Overview)

본 프로젝트는 오목(Gomoku) 환경에서 멀티에이전트 강화학습(MARL)을 적용하는 것을 목표로 합니다. 오목은 흑(선공)이 압도적으로 유리한 비대칭 게임입니다.

본 프로젝트는 `hesic73/gomoku_rl`을 기반으로, **흑(공격자) 에이전트**와 **백(방어자) 에이전트**를 독립적으로 학습시켜 , **PPO와 A2C 알고리즘의 학습 효율성과 전략적 차이를 비교 분석**합니다.

---

## 2. 🛠️ Tech Stack

* **Core:** Python 3.11.5, PyTorch 2.1.0, TorchRL 0.2.1
* **Environment:** Conda (로컬 가상 환경), Google Colab (GPU 학습)
* **Config & Demo:** Hydra-core, OmegaConf, PyQt5
* **Utilities:** NumPy (<2.0), WandB, Matplotlib

---

## 3. 🚀 프로젝트 실행 방법 (Getting Started)

팀원 모두 CPU 환경이므로, "개발은 로컬(CPU), 학습은 Colab(GPU)" 워크플로우를 따릅니다.

### 1. 저장소 복제 (Clone)

```bash
git clone [https://github.com/KWONSEOK02/Asymmetric_Gomoku_MARL.git](https://github.com/KWONSEOK02/Asymmetric_Gomoku_MARL.git)
cd Asymmetric_Gomoku_MARL
```


# 2. 로컬 가상 환경 설정 (Conda)

## Python 3.11.5로 'gomoku_marl_env' 환경 생성
conda create -n gomoku_marl_env python=3.11.5
conda activate gomoku_marl_env

## requirements.txt로 모든 라이브러리 설치
pip install -r requirements.txt


## 3. Colab을 이용한 GPU 학습 (Training)
    ### 1.Colab 노트북을 열고 런타임 유형을 GPU로 변경합니다.

    ### 2.Google Drive를 마운트(연결)합니다.
    from google.colab import drive
    drive.mount('/content/drive')


    ### 3. 프로젝트를 클론하고 라이브러리를 설치
        ###  GitHub에서 프로젝트를 복제합니다. (마크다운 없이 URL만 사용)
        !git clone https://github.com/KWONSEOK02/Asymmetric_Gomoku_MARL.git

        ###  복제된 폴더 안으로 작업 디렉터리를 이동합니다.
        %cd Asymmetric_Gomoku_MARL

        ###  프로젝트 폴더 안에서 requirements.txt 파일로 라이브러리를 설치합니다.
        !pip install -r requirements.txt

    # (필요시) Colab용 CUDA PyTorch 설치 (requirements.txt에 +cu118 추가 권장)
    # !pip install torch==2.1.0+cu118 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

    ### 4.Google Drive에 결과를 저장하며 학습을 실행합니다. (12시간 런타임 제한 대비)
        !python Asymmetric_Gomoku_MARL/scripts/train.py \
        device=cuda \
        wandb.mode=disabled \
        run_dir='/content/drive/My Drive/Gomoku_Results' \
        epochs=1000


## 4. 로컬에서 데모 실행 (Demo)
    ### 1. Google Drive의 Gomoku_Results 폴더에서 학습된 모델(예: black_final.pt)을 로컬 results/models/ 폴더로 다운로드합니다.

    ### 2. 로컬의 (gomoku_marl_env) 환경에서 demo.py를 실행합니다.

    # (gomoku_marl_env)
    python scripts/demo.py device=cpu checkpoint=results/models/black_final.pt


# 4. 📁 프로젝트 구조
Asymmetric_Gomoku_MARL/
├── configs/             # (신규) 실험 설정 (.yaml 파일)
├── data/                # [템플릿] 원본 데이터셋 (Raw, Processed)
│   ├── processed/
│   └── raw/
├── notebooks/           # [템플릿] 데이터 탐색(EDA), Colab/로컬 테스트용 노트북
├── results/             # [템플릿] 실험 결과물
│   ├── models/          # (Git 무시) 학습 완료된 모델 파일 (.pt, .pth)
│   └── plots/           # (Git 무시) Elo 그래프, 승률표
├── scripts/             # (신규) 실행 스크립트
│   ├── train.py         # (신규) 학습 실행
│   ├── evaluate.py      # (신규) 평가 실행 (흑 vs 백 Baseline 대결)
│   └── demo.py          # (신규) GUI 데모 실행
├── src/                 # [템플릿] 모든 파이썬 소스 코드 (핵심 로직)
│   ├── __init__.py
│   ├── agents/          # [템플릿] 에이전트 알고리즘 (PPO, A2C)
│   │   ├── __init__.py
│   │   ├── collectors.py  # (가져옴) BlackPlayCollector 등
│   │   ├── ppo_agent.py
│   │   └── a2c_agent.py
│   ├── envs/            # [템플릿] 커스텀 학습 환경
│   │   ├── __init__.py
│   │   ├── gomoku_env.py  # (가져옴) GomokuEnv 클래스
│   │   └── core.py        # (가져옴) Gomoku 핵심 로직 (승리 판정 등)
│   ├── evaluation/      # (신규) 성능 평가 로직
│   │   ├── __init__.py
│   │   └── evaluator.py   # (신규) 1:1 대결 및 보상/승률 계산 클래스
│   ├── models/          # [템플릿] 신경망 모델 구조 (설계도 .py)
│   │   ├── __init__.py
│   │   └── ppo_model.py
│   └── utils/           # [템플릿] 공통 유틸리티 (augment.py, log.py, policy.py)
│       └── __init__.py
├── .gitignore           # [템플릿] Git 무시 목록 (results/, data/, .ipynb_checkpoints/ 등)
├── requirements.txt     # [템플릿] 필요 라이브러리 목록 (pip install -r ...)
└── README.md            # [템플릿] 프로젝트 설명서 (본 파일)


# 5. 🤝 협업 가이드라인 (Contribution Guidelines)
    ## Git Workflow
        master (Production): 최종 배포 브랜치

        develop (Staging): 개발 완료 코드를 병합하는 메인 브랜치

        feat/*, fix/*, docs/*: 기능별, 목적별 브랜치

    ## 작업 흐름:

        develop 브랜치에서 feat/my-new-feature 브랜치를 생성하여 작업을 시작합니다.

        기능 개발 완료 후, develop 브랜치로 **Pull Request (PR)**를 생성합니다.

        팀원 1명 이상의 **코드 리뷰 및 승인(Approve)**을 받습니다.

        병합(Merge) 전, develop의 최신 변경 사항을 자주 동기화(pull)하여 충돌을 최소화합니다.

    ## 프로젝트 규칙
        PR은 작은 단위로: 하나의 PR은 하나의 기능에만 집중합니다.

        작업 완료 처리: 세부 작업은 체크리스트로, 큰 작업은 "Projects" 탭에서 관리합니다.

        충돌 최소화: 회의에서 작업 부분을 명확히 나누고, 다른 팀원의 작업이 필요한 부분은 미리 공유합니다.

    ## 개발 가이드라인
        코딩 스타일: PEP 8 준수, Linter(Flake8) 사용을 권장합니다.

        주석: 모든 함수와 클래스에 Google 스타일 Docstring을 작성합니다.

        환경 통일: 모든 팀원은 requirements.txt를 통해 동일한 라이브러리 버전을 사용합니다.

        커밋 및 브랜치 컨벤션
        Git 브랜치 및 커밋 메시지는 Conventional Commits 규칙을 준수합니다.

        (예: feat(agent): A2C 에이전트 알고리즘 추가, fix(reward): 보상 함수 계산 오류 수정)

        상세 규칙은 팀 Notion 페이지 링크: (https://www.notion.so/27167d3af687803ca8c1ec0a66bbeb59?source=copy_link)