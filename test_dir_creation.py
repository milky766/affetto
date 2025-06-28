from pathlib import Path
import datetime
import time

def create_test_directory():
    # apps/simulate_adaptive_control.py と同じロジックでパスを構築
    base_dir = Path("data")
    app_name = "test_simulation_output"
    label = "temp_test"
    
    # 現在のタイムスタンプを取得
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%dT%H%M%S")
    
    # フルパスを構築
    output_dir_path = base_dir / app_name / label / timestamp_str
    
    print(f"Attempting to create directory: {output_dir_path}")
    
    try:
        # ディレクトリを作成 (parents=True で親ディレクトリも作成、exist_ok=True で既存でもエラーにしない)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Successfully created directory: {output_dir_path}")
    except Exception as e:
        print(f"Error creating directory: {e}")

if __name__ == "__main__":
    create_test_directory()