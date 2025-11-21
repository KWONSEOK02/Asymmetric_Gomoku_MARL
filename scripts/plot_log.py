import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import sys

def parse_log(log_path):
    """
    로그 파일을 읽어 DataFrame으로 변환하는 함수
    """
    if not os.path.exists(log_path):
        print(f"❌ 오류: 파일을 찾을 수 없습니다. 경로를 확인하세요: {log_path}")
        return pd.DataFrame()

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    data = []
    current_epoch = 0
    
    # 임시 저장 변수
    bw, ww, fps = None, None, None

    for line in lines:
        if "Epoch" in line and "Results" in line:
            m = re.search(r"Epoch:(\d+)", line)
            if m: current_epoch = int(m.group(1))
            bw, ww, fps = None, None, None
        
        if "Black Win Rate:" in line:
            m = re.search(r"([0-9.]+)%", line)
            if m: bw = float(m.group(1))
        
        if "White Win Rate:" in line:
            m = re.search(r"([0-9.]+)%", line)
            if m: ww = float(m.group(1))

        if "FPS" in line:
             m = re.search(r"FPS.*:\s*([0-9.]+)", line)
             if m: fps = float(m.group(1))

        if bw is not None and ww is not None:
            data.append({
                'epoch': current_epoch, 
                'black_win': bw, 
                'white_win': ww,
                'fps': fps if fps else 0
            })
            bw, ww = None, None

    return pd.DataFrame(data)

def print_summary(df):
    """
    보고서 작성용 요약 테이블 출력
    """
    if df.empty: return

    df = df.sort_values("epoch").reset_index(drop=True)
    start = df.iloc[0]
    mid = df.iloc[len(df)//2]
    end = df.iloc[-1]

    print(f"\n📊 [학습 결과 요약] ---------------------------------------")
    print(f"{'시점':<10} | {'Epoch':<6} | {'Black Win':<10} | {'White Win':<10}")
    print(f"----------------------------------------------------------")
    print(f"{'시작(Start)':<10} | {int(start['epoch']):<6} | {start['black_win']:>6.2f}%    | {start['white_win']:>6.2f}%")
    print(f"{'중간(Mid)':<10} | {int(mid['epoch']):<6} | {mid['black_win']:>6.2f}%    | {mid['white_win']:>6.2f}%")
    print(f"{'최종(End)':<10} | {int(end['epoch']):<6} | {end['black_win']:>6.2f}%    | {end['white_win']:>6.2f}%")
    print(f"----------------------------------------------------------\n")

def plot_graph(df, save_path, title_name):
    """
    그래프를 그리고 지정된 경로에 이미지로 저장하는 함수
    (파일명에 따라 주인공 색상 강조)
    """
    if df.empty:
        print("⚠️ 데이터가 없어 그래프를 그릴 수 없습니다.")
        return
    plt.figure(figsize=(10, 6))
    
    # --- 팀 규칙 적용: 흑(Orange), 백(Blue), 실선(-) ---
    plt.plot(df['epoch'], df['black_win'], label='Black Win Rate', 
             color='orange', linestyle='-', linewidth=2.5)
    
    plt.plot(df['epoch'], df['white_win'], label='White Win Rate', 
             color='royalblue', linestyle='-', linewidth=2.5)
    # ----------------------------------------------------
    
    plt.title(f"Training Progress: {title_name}", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Win Rate (%)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 이미지 저장
    plt.savefig(save_path, dpi=100)
    print(f"✅ 그래프 이미지가 저장되었습니다: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="로그 파일을 분석하여 그래프를 그립니다.")
    parser.add_argument("log_path", type=str, help="분석할 train.log 파일의 경로")
    args = parser.parse_args()

    log_path = args.log_path
    
    # 1. 파일 이름(확장자 제외) 추출 (예: Collapse_PA_A2C_white_v2)
    log_name = os.path.splitext(os.path.basename(log_path))[0]

    # 2. 저장 경로 설정 (우선순위: results/plots -> 없으면 로그 파일 옆)
    # 현재 실행 위치 기준으로 results/plots 폴더가 있는지 확인
    preferred_plot_dir = os.path.join("results", "plots")
    
    if os.path.exists(preferred_plot_dir):
        save_dir = preferred_plot_dir
    else:
        # plots 폴더가 없으면 그냥 로그 파일 옆에 저장
        save_dir = os.path.dirname(log_path)
    
    # 파일명 겹침 방지를 위해 로그 파일 이름을 그대로 이미지 이름으로 사용
    save_path = os.path.join(save_dir, f"{log_name}.png")

    print(f"📂 분석 시작: {log_path}")

    # 3. 파싱 및 실행
    df = parse_log(log_path)
    print_summary(df)
    plot_graph(df, save_path, log_name)

if __name__ == "__main__":
    main()