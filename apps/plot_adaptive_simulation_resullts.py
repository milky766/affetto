# apps/plot_adaptive_simulation_results.py

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # np.nan のために追加

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot results from adaptive control simulation.")
    parser.add_argument("results_csv", type=Path, help="Path to the adaptive_simulation_results.csv file.")
    parser.add_argument("-o", "--output", type=Path, help="Output directory for plots. Defaults to results_csv parent.")
    parser.add_argument("--show-screen", action="store_true", help="Display plots on screen.")
    # プロット対象の関節IDをコマンドライン引数で受け取る
    parser.add_argument("--plot-joints", nargs="+", type=int, help="Specific joint IDs to plot. Defaults to all joints found in data.")
    args = parser.parse_args()

    results_path: Path = args.results_csv
    if not results_path.exists():
        print(f"Error: Results CSV file not found at {results_path}")
        return

    output_dir: Path = args.output if args.output else results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_path)

    # データフレームの列名からアクティブな関節IDを特定
    # current_q_jX の形式を期待
    all_joint_cols = [col for col in df.columns if col.startswith('current_q_j')]
    if not all_joint_cols:
        print("Error: No joint position columns (e.g., 'current_q_jX') found in CSV.")
        return

    # 実際にプロットする関節IDのリストを決定
    # コマンドライン引数で指定がなければ、見つかったすべての関節をプロット
    active_joints_in_data = sorted([int(col.split('_j')[1]) for col in all_joint_cols])
    
    joints_to_plot = args.plot_joints if args.plot_joints else active_joints_in_data
    
    if not joints_to_plot:
        print("No joints to plot. Exiting.")
        return

    print(f"Plotting results for joints: {joints_to_plot}")

    for joint_idx in joints_to_plot:
        # --- 関節位置追従のプロット (current_q_jX vs ref_q_current_jX) ---
        joint_q_col = f"current_q_j{joint_idx}"
        ref_q_col = f"ref_q_current_j{joint_idx}"

        if joint_q_col in df.columns and ref_q_col in df.columns:
            fig, ax1 = plt.subplots(figsize=(12, 6))

            ax1.plot(df['time'], df[joint_q_col], label=f'Joint {joint_idx} Actual Q')
            ax1.plot(df['time'], df[ref_q_col], label=f'Joint {joint_idx} Reference Q', linestyle='--')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Joint Position', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_title(f'Joint {joint_idx} Position Tracking & Selected Tau')
            ax1.grid(True)
            ax1.legend(loc='upper left')

            # 選択された Tau を同じグラフの第二Y軸にプロット
            if 'selected_tau' in df.columns:
                ax2 = ax1.twinx()
                ax2.plot(df['time'], df['selected_tau'], label='Selected Tau', color='red', linestyle=':')
                ax2.set_ylabel('Selected Tau (Preview Steps)', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                ax2.legend(loc='upper right')

            plt.tight_layout()
            plt.savefig(output_dir / f'joint_{joint_idx}_position_tracking_with_tau.png')
            if args.show_screen:
                plt.show()
            plt.close(fig) # 必ず閉じる


        # --- 制御入力予測のプロット (predicted_ca/cb_jX_at_tau vs true_ca/cb_jX_at_tau) ---
        pred_ca_col = f"predicted_ca_j{joint_idx}_at_tau"
        true_ca_col = f"true_ca_j{joint_idx}_at_tau"
        pred_cb_col = f"predicted_cb_j{joint_idx}_at_tau"
        true_cb_col = f"true_cb_j{joint_idx}_at_tau"

        if (pred_ca_col in df.columns and true_ca_col in df.columns and
            pred_cb_col in df.columns and true_cb_col in df.columns):
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            ax1.plot(df['time'], df[pred_ca_col], label=f'Ca (pred)', color='blue')
            ax1.plot(df['time'], df[true_ca_col], label=f'Ca (true)', color='blue', linestyle='--')
            ax1.plot(df['time'], df[pred_cb_col], label=f'Cb (pred)', color='orange')
            ax1.plot(df['time'], df[true_cb_col], label=f'Cb (true)', color='orange', linestyle='--')

            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Pressure at Valve [kPa]')
            ax1.set_title(f'Joint {joint_idx} Control Input Prediction')
            ax1.legend()
            ax1.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / f'joint_{joint_idx}_control_input_prediction.png')
            if args.show_screen:
                plt.show()
            plt.close(fig) # 必ず閉じる


    # --- R^2 スコアのプロット ---
    # 各τごとのR^2列を動的に収集
    r2_tau_cols = [col for col in df.columns if col.startswith('r2_tau_')]
    if r2_tau_cols:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for col in r2_tau_cols:
            tau_val = col.split('_')[-1] # 'r2_tau_X' から X を取得
            ax.plot(df['time'], df[col], label=f'R^2 for τ={tau_val}')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('R^2 Score')
        ax.set_title('R^2 Scores for Each Tau over Time')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / 'r2_scores_over_time.png')
        if args.show_screen:
            plt.show()
        plt.close(fig) # 必ず閉じる


if __name__ == "__main__":
    main()

