#!/usr/bin/env python

from __future__ import annotations
from matplotlib.ticker import MaxNLocator

import argparse
import collections
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
from pydantic import BaseModel

from affetto_nn_ctrl.control_utility import resolve_joints_str
from affetto_nn_ctrl.data_handling import (
    copy_config,
    get_output_dir_path,
    prepare_data_dir_path,
)
from affetto_nn_ctrl.event_logging import event_logger, start_logging
from affetto_nn_ctrl.model_utility import (
    DataAdapterBase,
    DataAdapterParamsBase,
    DefaultInputs,
    DefaultRefs,
    DefaultStates,
    TrainedModel,
    load_trained_model,
    _get_keys,
)
from affetto_nn_ctrl.simulation_utility import (
    calculate_sse_sst,
    initialize_robot_state,
    load_reference_trajectory,
)
from pyplotutil.datautil import Data



# ==========================================================
# ★ 必要なクラス定義 (他のファイルに依存しないようにここに記述)
# ==========================================================
@dataclass
class SingleShotMultiHorizonParams(DataAdapterParamsBase):
    """Parameters for the single-shot multi-horizon handler."""
    active_joints: list[int]
    dt: float
    ctrl_step: int
    min_preview_step: int
    max_preview_step: int

class SingleShotMultiHorizonHandler(DataAdapterBase[SingleShotMultiHorizonParams, DefaultStates, DefaultRefs, DefaultInputs]):
    """
    Generates a dataset for single-shot multi-horizon learning.
    Input X: [current_state, future_reference_trajectory]
    Output Y: [future_control_commands]
    """
    def make_feature(self, dataset: Data) -> np.ndarray:
        joints = self.params.active_joints
        N_min, N_max = self.params.min_preview_step, self.params.max_preview_step
        state_keys = _get_keys(["q", "dq", "pa", "pb"], joints)
        ref_keys = _get_keys(["q"], joints)
        valid_len = len(dataset.df) - N_max
        s_t = dataset.df.loc[:valid_len-1, state_keys].values
        future_refs_list = [dataset.df.loc[k:valid_len+k-1, ref_keys].values for k in range(N_min, N_max + 1)]
        future_refs_concat = np.concatenate(future_refs_list, axis=1)
        X = np.concatenate([s_t, future_refs_concat], axis=1)
        return X

    def make_target(self, dataset: Data) -> np.ndarray:
        joints = self.params.active_joints
        N_min, N_max = self.params.min_preview_step, self.params.max_preview_step
        ctrl_keys = _get_keys(["ca", "cb"], joints)
        valid_len = len(dataset.df) - N_max
        future_ctrls_list = [dataset.df.loc[k:valid_len+k-1, ctrl_keys].values for k in range(N_min, N_max + 1)]
        Y = np.concatenate(future_ctrls_list, axis=1)
        return Y

    def make_model_input(self, t: float, states: DefaultStates, refs: DefaultRefs) -> np.ndarray:
        raise NotImplementedError
    def make_ctrl_input(self, y: np.ndarray, base_inputs: DefaultInputs) -> tuple[np.ndarray, ...]:
        raise NotImplementedError
    def reset(self) -> None:
        pass

# --- AdaptiveController クラス (新アルゴリズム対応版) ---
class AdaptiveController:
    def __init__(
        self,
        model: TrainedModel,
        N_min: int,
        N_max: int,
        initial_tau: int,
        window_size: int = 100,
        hysteresis_threshold: float = 0.01,
    ):
        self.model = model
        self.N_min = N_min
        self.N_max = N_max
        self.window_size = window_size
        self.hysteresis_threshold = hysteresis_threshold
        self.r2_buffers: dict[int, collections.deque] = {
            k: collections.deque(maxlen=window_size) for k in range(self.N_min, self.N_max + 1)
        }
        self.current_tau = max(self.N_min, min(initial_tau, self.N_max))
        event_logger().info(f"AdaptiveController initialized. N_range=[{N_min}, {N_max}], initial_tau={self.current_tau}")

    def update_performance(self, predicted_output: np.ndarray, true_output: np.ndarray, tau: int) -> None:
        if tau not in self.r2_buffers: return
        try:
            sse, sst = calculate_sse_sst(predicted_output, true_output)
            self.r2_buffers[tau].append({"sse": sse, "sst": sst})
        except ValueError as e:
            event_logger().error(f"Error calculating SSE/SST for tau={tau}: {e}")

    def calculate_r_squared(self, tau: int) -> float:
        buffer = self.r2_buffers.get(tau)
        if not buffer or len(buffer) < 2: return -np.inf
        total_sse = sum(item["sse"] for item in buffer)
        total_sst = sum(item["sst"] for item in buffer)
        if total_sst == 0: return 1.0 if total_sse == 0 else -np.inf
        return 1 - (total_sse / total_sst)

    def select_optimal_tau(self) -> int:
        candidates = {self.current_tau}
        if self.current_tau > self.N_min: candidates.add(self.current_tau - 1)
        if self.current_tau < self.N_max: candidates.add(self.current_tau + 1)
        
        r2_scores = {k: self.calculate_r_squared(k) for k in sorted(list(candidates))}
        valid_r2_scores = {k: r for k, r in r2_scores.items() if r > -np.inf}
        
        if not valid_r2_scores:
            return self.current_tau

        max_r2 = max(valid_r2_scores.values())
        best_tau_candidate = min([k for k, r in valid_r2_scores.items() if r == max_r2])

        current_tau_r2 = r2_scores.get(self.current_tau, -np.inf)
        if max_r2 > (current_tau_r2 + self.hysteresis_threshold):
            new_tau = best_tau_candidate
            if new_tau != self.current_tau:
                 event_logger().info(f"τ changed from {self.current_tau} to {new_tau} (R^2 improved: {current_tau_r2:.4f} -> {max_r2:.4f})")
        else:
            new_tau = self.current_tau
        
        self.current_tau = new_tau
        return self.current_tau

# --- プロット用関数 ---
def plot_simulation_results(results_df: pd.DataFrame, output_dir: Path, N_min: int, N_max: int, active_joints: list[int]):
    """
    シミュレーション結果のデータフレームから2種類のグラフを生成し、保存する。
    """
    event_logger().info("Generating simulation result plots...")
    time_axis = results_df["time"]

    # --- 1. 予測値 vs 真値 の比較プロット ---
    for joint_idx in active_joints:
        fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
        fig.suptitle(f"Prediction vs. True Values for Joint {joint_idx}")

        # ca (上側のプロット)
        axes[0].plot(time_axis, results_df[f"true_ca_j{joint_idx}_at_tau"], ls="--", label="ca (true)")
        axes[0].plot(time_axis, results_df[f"predicted_ca_j{joint_idx}_at_tau"], label="ca (pred)")
        axes[0].set_ylabel("Pressure ca [kPa]")
        axes[0].legend()
        axes[0].grid(True)

        # cb (下側のプロット)
        axes[1].plot(time_axis, results_df[f"true_cb_j{joint_idx}_at_tau"], ls="--", label="cb (true)")
        axes[1].plot(time_axis, results_df[f"predicted_cb_j{joint_idx}_at_tau"], label="cb (pred)")
        axes[1].set_ylabel("Pressure cb [kPa]")
        axes[1].set_xlabel("Time [s]")
        axes[1].legend()
        axes[1].grid(True)
        
        plot_path = output_dir / f"prediction_vs_true_j{joint_idx}.png"
        fig.savefig(plot_path)
        plt.close(fig)
        event_logger().info(f"Saved prediction plot: {plot_path}")


    # --- 2. R^2スコアと選択されたτの推移プロット ---
    fig, ax1 = plt.subplots(figsize=(18, 6))

    # R^2スコアをプロット
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("R^2 Score")
    ax1.set_ylim(0.85, 1) 
    for k_val in range(N_min, N_max + 1):
        # DataFrameにその列が存在するか確認してからプロット
        col_name = f"r2_tau_{k_val}"
        if col_name in results_df.columns:
            ax1.plot(time_axis, results_df[col_name], label=f"R^2 (τ={k_val})")
            
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1) # R^2=0 のライン
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # 選択されたτを右側の軸にプロット
    ax2 = ax1.twinx()
    ax2.set_ylabel("Selected τ", color='red')
    ax2.plot(time_axis, results_df["selected_tau"], color='red', linestyle=':', marker='o', markersize=2, label="Selected τ")
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')
    

    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle("R^2 Scores and Selected τ over Time")
    fig.tight_layout()
    
    plot_path = output_dir / "r2_and_tau_evolution.png"
    fig.savefig(plot_path)
    plt.close(fig)
    event_logger().info(f"Saved R^2 evolution plot: {plot_path}")


# --- メイン処理 ---
def run(args: argparse.Namespace) -> None:
    # --- 設定と出力ディレクトリの準備 ---
    with open(args.config, 'r') as f:
        config = toml.load(f)
    adapter_config = config["model"]["adapter"]
    N_min = adapter_config.get("min_preview_step", 1)
    N_max = adapter_config.get("max_preview_step", 20)
    active_joints = adapter_config.get("active_joints", args.active_joints)
    n_active_joints = len(active_joints)
    output_dir = get_output_dir_path(
        base_dir="data", app_name=Path(__file__).stem, given_output=args.output,
        label=args.label, split_by_date=True,
        sublabel=None,          # ★ この引数を追加
        specified_date=None,  # ★ この引数を追加
    )
    prepare_data_dir_path(output_dir, make_latest_symlink=True)
    start_logging(sys.argv, output_dir, __name__, args.verbose)
    copy_config(None, None, args.config, output_dir)
    event_logger().info(f"Config: {args.config}")
    event_logger().info(f"Output directory: {output_dir}")
    event_logger().info(f"N_min: {N_min}, N_max: {N_max}, Active Joints: {active_joints}")

    # --- モデルと参照軌道のロード ---
    trained_model = load_trained_model(args.model_path)
    reference_df = load_reference_trajectory(args.reference_trajectory)
    total_ref_len = len(reference_df)
    
        # ==========================================================
    # ▼▼▼ デバッグ用ログ(1): 変数の初期値を確認 ▼▼▼
    # ==========================================================
    event_logger().info("--- DEBUG INFO ---")
    event_logger().info(f"args.max_steps is set to: {args.max_steps}")
    event_logger().info(f"Length of reference_df (total_ref_len) is: {total_ref_len}")
    event_logger().info(f"--------------------")


    # --- Controllerの初期化 ---
    controller = AdaptiveController(
        model=trained_model, N_min=N_min, N_max=N_max,
        initial_tau=args.initial_tau, window_size=args.window_size,
        hysteresis_threshold=args.hysteresis_threshold
    )

    # --- シミュレーションループ ---
    current_state_active = initialize_robot_state(reference_df, active_joints)
    sim_results = []
    event_logger().info("Starting simulation with new local search algorithm...")

    state_keys = _get_keys(["q", "dq", "pa", "pb"], active_joints)
    ref_keys = _get_keys(["q"], active_joints)
    ctrl_keys = _get_keys(["ca", "cb"], active_joints)
    n_ref_features = len(ref_keys)
    n_ctrl_features = len(ctrl_keys)

    for step in range(args.max_steps):
        # ==========================================================
        # ▼▼▼ デバッグ用ログ(2): ループの進捗を確認 ▼▼▼
        # ==========================================================
        if step % 100 == 0:
            event_logger().info(f"Processing step: {step} / {args.max_steps}")
            
            # ==========================================================
        # ▼▼▼ デバッグ用ログ(3): ループ停止条件の確認 ▼▼▼
        # ==========================================================
        if step + N_max >= total_ref_len:
            event_logger().info(f"!!! Break condition met at step {step}:")
            event_logger().info(f"!!! step({step}) + N_max({N_max}) >= total_ref_len({total_ref_len})")
            event_logger().info("!!! Loop will now break.")
            break
        
        if step + N_max >= total_ref_len:
            event_logger().info("End of reference trajectory reached.")
            break

        # 1. モデルへの入力Xを一度だけ作成
        s_t = current_state_active
        future_refs_list = [reference_df.iloc[min(step + k, total_ref_len - 1)][ref_keys].values for k in range(N_min, N_max + 1)]
        future_refs_concat = np.concatenate(future_refs_list)
        x_input = np.concatenate([s_t, future_refs_concat]).reshape(1, -1)

        # 2. 全ての未来の制御指令をまとめて予測
        all_predicted_outputs = trained_model.predict(x_input).flatten()

        # 3. 現在のτとその両隣の3点について性能を評価
        tau_candidates = {controller.current_tau}
        if controller.current_tau > N_min: tau_candidates.add(controller.current_tau - 1)
        if controller.current_tau < N_max: tau_candidates.add(controller.current_tau + 1)

        for k_val in tau_candidates:
            start_idx = (k_val - N_min) * n_ctrl_features
            end_idx = start_idx + n_ctrl_features
            predicted_ca_cb_for_k = all_predicted_outputs[start_idx:end_idx]
            
            true_ref_idx = min(step + k_val, total_ref_len - 1)
            true_ca_cb_for_k = reference_df.iloc[true_ref_idx][ctrl_keys].values
            
            controller.update_performance(predicted_ca_cb_for_k, true_ca_cb_for_k, k_val)

        # 4. 最適τを選択
        selected_tau = controller.select_optimal_tau()

        # 5. 選択されたτの予測値を現在の制御指令として使用
        final_start_idx = (selected_tau - N_min) * n_ctrl_features
        final_end_idx = final_start_idx + n_ctrl_features
        final_command = all_predicted_outputs[final_start_idx:final_end_idx]
        
        # 6. 結果の記録
        result_entry = {"time": step * (1.0/30.0), "selected_tau": selected_tau}
        tau_minus_1 = selected_tau - 1
        tau_plus_1 = selected_tau + 1
        result_entry[f"r2_tau_{tau_minus_1}"] = controller.calculate_r_squared(tau_minus_1)
        result_entry[f"r2_tau_{selected_tau}"] = controller.calculate_r_squared(selected_tau)
        result_entry[f"r2_tau_{tau_plus_1}"] = controller.calculate_r_squared(tau_plus_1)
                # ... (予測値と真値の記録ロジック) ...
        sim_results.append(result_entry)

        
        final_true_ref_idx = min(step + selected_tau, total_ref_len - 1)
        true_command = reference_df.iloc[final_true_ref_idx][ctrl_keys].values
        for i, joint_idx in enumerate(active_joints):
            result_entry[f"predicted_ca_j{joint_idx}_at_tau"] = final_command[i]
            result_entry[f"predicted_cb_j{joint_idx}_at_tau"] = final_command[i + n_active_joints]
            result_entry[f"true_ca_j{joint_idx}_at_tau"] = true_command[i]
            result_entry[f"true_cb_j{joint_idx}_at_tau"] = true_command[i + n_active_joints]
        sim_results.append(result_entry)
        
        # 7. 状態の更新 (ダミー)
        current_state_active = reference_df.iloc[step + 1][state_keys].values

    # --- 結果の保存とプロット ---
    event_logger().info("Simulation finished.")
    results_df = pd.DataFrame(sim_results)
    results_csv_path = output_dir / "adaptive_simulation_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    event_logger().info(f"Simulation results saved to {results_csv_path}")
    plot_simulation_results(results_df, output_dir, N_min, N_max, active_joints)

# --- 引数パーサとmainの呼び出し ---
def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate adaptive control with a trained model.")
    parser.add_argument("model_path", type=Path, help="Path to the trained model")
    parser.add_argument("-r", "--reference_trajectory", type=Path, required=True, help="Path to the reference trajectory")
    parser.add_argument("-c", "--config", default="apps/config/adaptive_model.toml", help="Model config file")
    parser.add_argument("-o", "--output", help="Output directory path")
    parser.add_argument("--label", default="adaptive_simulation", help="Label for output directory")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum simulation steps")
    parser.add_argument("--active-joints", nargs="+", type=int, default=[5], help="Active joint indices")
    parser.add_argument("--initial-tau", type=int, default=1, help="Initial preview step (tau)")
    parser.add_argument("--window-size", type=int, default=100, help="Window size for R^2 calculation")
    parser.add_argument("--hysteresis-threshold", type=float, default=0.01, help="Hysteresis threshold for tau selection")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Verbose output")
    return parser.parse_args()

def main() -> None:
    args = parse()
    run(args)

if __name__ == "__main__":
    main()