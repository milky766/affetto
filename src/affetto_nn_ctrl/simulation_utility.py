from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

def load_reference_trajectory(path: Path) -> pd.DataFrame:
    """
    参照軌道データ（CSVファイル）をロードします。
    このファイルには、時刻 't'、関節角度 'qX'、速度 'dqX'、外力 'paX', 'pbX'、制御入力 'caX', 'cbX' などが
    含まれることを期待します。pandas DataFrameとしてそのまま返します。
    """
    if not path.exists():
        raise FileNotFoundError(f"Reference trajectory file not found: {path}")
    
    return pd.read_csv(path)

def initialize_robot_state(reference_df: pd.DataFrame, active_joints: list[int]) -> np.ndarray:
    """
    ロボットの初期状態 (q, dq, pa, pb) をアクティブな関節について、
    参照DataFrameの最初の行から取得して初期化します。
    MultiHorizonHandlerが期待する形式 ([q_active, dq_active, pa_active, pb_active]) に連結して返します。
    """
    if reference_df.empty:
        raise ValueError("Reference DataFrame is empty, cannot initialize robot state.")

    # 最初の時刻 t=0 のデータを取得
    initial_row = reference_df.iloc[0]
    
    # 使用するキーを動的に生成
    q_cols = [f"q{j}" for j in active_joints]
    dq_cols = [f"dq{j}" for j in active_joints]
    pa_cols = [f"pa{j}" for j in active_joints]
    pb_cols = [f"pb{j}" for j in active_joints]

    # 必要な列がDataFrameに存在しない場合、ゼロを埋める
    q_init = initial_row[q_cols].values if all(col in initial_row for col in q_cols) else np.zeros(len(active_joints))
    dq_init = initial_row[dq_cols].values if all(col in initial_row for col in dq_cols) else np.zeros(len(active_joints))
    pa_init = initial_row[pa_cols].values if all(col in initial_row for col in pa_cols) else np.zeros(len(active_joints))
    pb_init = initial_row[pb_cols].values if all(col in initial_row for col in pb_cols) else np.zeros(len(active_joints))
    
    # 連結
    initial_state_concatenated = np.concatenate([q_init, dq_init, pa_init, pb_init])
    return initial_state_concatenated

def run_simulation_step(
    current_idx: int, # シミュレーションの現在の参照軌道インデックス
    total_ref_len: int, # 参照軌道の全長
    reference_df: pd.DataFrame, # 全体の参照DataFrame
    active_joints: list[int],
    tau_val: int # The selected preview step k, needed for true_target_ca_cb_next_k
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    シミュレーションを1ステップ進めますが、ロボットの動きは既存のデータから読み込みます。
    
    Args:
        current_idx: シミュレーションの現在の参照軌道インデックス。
        total_ref_len: 参照軌道の総ステップ数。
        reference_df: `load_reference_trajectory`でロードされた完全な参照DataFrame。
        active_joints: アクティブな関節のインデックスリスト。
        tau_val: `tau`の値 (R^2計算用の真値の取得に使用)。

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - next_state_active: 次のステップでのロボットのアクティブ関節に関する状態 (q, dq, pa, pb) (データから取得)。
            - current_state_active_for_model_input: 現在のステップでのモデル入力用の状態 (q, dq, pa, pb)。
            - true_q_current: 現在のステップでの真の関節位置 (q) [アクティブ関節のみ]。
            - true_target_ca_cb_next_k: R^2計算用の、参照軌道から得られる真の未来のca, cb値。
    """
    n_active_joints = len(active_joints)

    # 次のステップのデータを取得z
    next_idx = min(current_idx + 1, total_ref_len - 1)
    current_row = reference_df.iloc[current_idx]
    next_row = reference_df.iloc[next_idx]

    # 現在のステップでのモデル入力用の状態を構築
    q_cols = [f"q{j}" for j in active_joints]
    dq_cols = [f"dq{j}" for j in active_joints]
    pa_cols = [f"pa{j}" for j in active_joints]
    pb_cols = [f"pb{j}" for j in active_joints]

    # 必要な列がDataFrameに存在しない場合、ゼロを埋める
    current_q = current_row[q_cols].values if all(col in current_row for col in q_cols) else np.zeros(n_active_joints)
    current_dq = current_row[dq_cols].values if all(col in current_row for col in dq_cols) else np.zeros(n_active_joints)
    current_pa = current_row[pa_cols].values if all(col in current_row for col in pa_cols) else np.zeros(n_active_joints)
    current_pb = current_row[pb_cols].values if all(col in current_row for col in pb_cols) else np.zeros(n_active_joints)
    current_state_active_for_model_input = np.concatenate([current_q, current_dq, current_pa, current_pb])

    # next_state_active (次のループでcurrent_state_activeになるもの)
    next_q = next_row[q_cols].values if all(col in next_row for col in q_cols) else np.zeros(n_active_joints)
    next_dq = next_row[dq_cols].values if all(col in next_row for col in dq_cols) else np.zeros(n_active_joints)
    next_pa = next_row[pa_cols].values if all(col in next_row for col in pa_cols) else np.zeros(n_active_joints)
    next_pb = next_row[pb_cols].values if all(col in next_row for col in pb_cols) else np.zeros(n_active_joints)
    next_state_active = np.concatenate([next_q, next_dq, next_pa, next_pb])

    # `true_q_current` は、現在のステップでのロボットの実際の関節位置 `q`
    true_q_current = current_q
    
    # --- R^2 計算用の真の未来の ca, cb 値 ---
    # `tau_val` ステップ先のインデックスを計算
    target_idx_for_r2 = min(current_idx + tau_val, total_ref_len - 1)
    
    # `ca` と `cb` の列名を動的に生成
    ca_cols = [f"ca{j}" for j in active_joints]
    cb_cols = [f"cb{j}" for j in active_joints]
    
    # 参照DataFrameから真の `ca`, `cb` を抽出
    true_ca_next_k = reference_df.iloc[target_idx_for_r2][ca_cols].values if all(col in reference_df.columns for col in ca_cols) else np.zeros(n_active_joints)
    true_cb_next_k = reference_df.iloc[target_idx_for_r2][cb_cols].values if all(col in reference_df.columns for col in cb_cols) else np.zeros(n_active_joints)
    
    true_target_ca_cb_next_k = np.concatenate([true_ca_next_k, true_cb_next_k])

    return next_state_active, current_state_active_for_model_input, true_q_current, true_target_ca_cb_next_k

def calculate_sse_sst(predicted_output: np.ndarray, true_output: np.ndarray) -> tuple[float, float]:
    """
    予測出力と真の出力からSSEとSSTを計算します。
    R^2 = 1 - (SSE / SST)
    """
    if predicted_output.shape != true_output.shape:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Shape mismatch in calculate_sse_sst: predicted_output {predicted_output.shape} vs true_output {true_output.shape}")
        raise ValueError(f"Shape mismatch: predicted_output {predicted_output.shape} vs true_output {true_output.shape}")

    sse = np.sum((true_output - predicted_output)**2)
    mean_true_output = np.mean(true_output)
    sst = np.sum((true_output - mean_true_output)**2)

    return float(sse), float(sst)
