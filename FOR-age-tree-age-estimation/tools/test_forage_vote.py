import os
import subprocess
import pandas as pd

CONFIG_FILE_PATH = '/workspace/configs/forestformer_age_pre.py'
BASE_OUTPUT_DIR = '/workspace/work_dirs/load_ff3d_forage/test_forage2'
MODEL_PATH = '/workspace/work_dirs/load_ff3d_forage/best_RMSE_epoch_1050_fixed.pth'
ROUND_NUM = 3
VOTE_NUM = 10
test_or_val = 'test'

def modify_config(test_output_dir):
    lines = []
    modified = False
    with open(CONFIG_FILE_PATH, 'r') as f:
        for line in f:
            if line.strip().startswith("test_output_dir = ") and not modified:
                lines.append(f"test_output_dir = '{test_output_dir}'\n")
                modified = True
            else:
                lines.append(line)
    if not modified:
        raise RuntimeError("test_output_dir not found in config!")
    with open(CONFIG_FILE_PATH, 'w') as f:
        f.writelines(lines)

def modify_config2(test_or_val):
    lines = []
    modified = False
    with open(CONFIG_FILE_PATH, 'r') as f:
        for line in f:
            if line.strip().startswith("test_or_val = ") and not modified:
                lines.append(f"test_or_val = '{test_or_val}'\n")
                modified = True
            else:
                lines.append(line)
    if not modified:
        raise RuntimeError("test_output_dir not found in config!")
    with open(CONFIG_FILE_PATH, 'w') as f:
        f.writelines(lines)

def run_test_script():
    command = f"CUDA_VISIBLE_DEVICES=0 python /workspace/tools/test.py {CONFIG_FILE_PATH} {MODEL_PATH}"
    subprocess.run(command, shell=True, check=True)

def average_votes(csv_paths):
    dfs = [pd.read_csv(p) for p in csv_paths]
    merged = pd.concat(dfs).groupby("treeID").agg({
        "predicted_age": "mean",
        "age_label": "first"
    }).reset_index()
    return merged

def run_plot(file_path):
    plot_path = file_path.replace(".csv", ".png")
    command = f"python /workspace/tools/plot_age_dots.py {file_path} {plot_path}"
    subprocess.run(command, shell=True)

def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    round_results = []

    for r in range(ROUND_NUM):
        round_dir = os.path.join(BASE_OUTPUT_DIR, f"round{r}")
        os.makedirs(round_dir, exist_ok=True)
        vote_csv_paths = []

        for v in range(VOTE_NUM):
            vote_dir = os.path.join(round_dir, f"vote{v}")
            os.makedirs(vote_dir, exist_ok=True)
            modify_config(vote_dir)
            modify_config2(test_or_val)
            run_test_script()
            if test_or_val == 'test':
                vote_csv_path = os.path.join(vote_dir, 'test.csv')
            elif test_or_val == 'val':
                vote_csv_path = os.path.join(vote_dir, 'val.csv')
            vote_csv_paths.append(vote_csv_path)

        round_df = average_votes(vote_csv_paths)
        round_csv_path = os.path.join(BASE_OUTPUT_DIR, f"round_{r}_mean.csv")
        round_df.to_csv(round_csv_path, index=False, float_format='%.2f')
        run_plot(round_csv_path)
        round_results.append(round_df)

    # Final summary
    all_df = pd.concat(round_results)
    final_df = all_df.groupby("treeID").agg({
        "predicted_age": ["mean", "std"],
        "age_label": "first"
    })
    final_df.columns = ['predicted_age_mean', 'predicted_age_std', 'age_label']
    final_df = final_df.reset_index()

    final_summary_path = os.path.join(BASE_OUTPUT_DIR, 'final_summary.csv')
    final_df.to_csv(final_summary_path, index=False, float_format='%.2f')
    run_plot(final_summary_path)

if __name__ == "__main__":
    main()
