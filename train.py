# train.py

import subprocess
import os
import time
from pathlib import Path
import glob
import replace
import pandas as pd

# ─── Your existing helpers ────────────────────────────────────────────────
def get_next_run_number(base_name, results_dir="./results"):
    """Get the next run number for a given base name by checking existing results."""
    os.makedirs(results_dir, exist_ok=True)
    pattern = os.path.join(results_dir, f"{base_name}_*")
    existing_runs = glob.glob(pattern)
    if not existing_runs:
        return 1
    run_numbers = []
    for run_path in existing_runs:
        try:
            run_num = int(run_path.split('_')[-1])
            run_numbers.append(run_num)
        except (ValueError, IndexError):
            continue
    return max(run_numbers) + 1 if run_numbers else 1

def summarize_log(log_path: str):
    """
    Reads the Unity log at log_path, then prints:
      • Overall success rate (%)
      • Success rate per trial type
      • Max target distance (units)
    """
    # 1) load into DataFrame
    df = pd.read_csv(
        log_path,
        sep=r'\s+',                # whitespace-separated
        comment='#',               # skip any comment lines
        header=None,
        names=['SessionTime','EventType','x','y','z','r','extra'],
        usecols=[0,1,2,4,5],       # we only need x, z for distance
        engine='python'
    )
    # only keep the “n”, “t”, “h”, “f” events
    df = df[df.EventType.isin(['n','t','s','h','f'])].reset_index(drop=True)

    # 2) find indices of each new trial
    new_trial_idxs = df.index[df.EventType=='n'].tolist()
    trial_type_idx = df.index[df.EventType=='s'].tolist()

    successes = []
    by_type = {}
    distances = []

    for ti, start_idx in enumerate(new_trial_idxs):
        end_idx = new_trial_idxs[ti+1] if ti+1 < len(new_trial_idxs) else len(df)
        trial = df.iloc[start_idx:end_idx]

        # trial type code is in the 'x' column of the 'n' line
        ttype = int(trial.loc[trial.EventType=='s','x'].iat[0])

        # find the target distance (first 't' row)
        trow = trial[trial.EventType=='t']
        if trow.empty:
            continue
        dx, dz = float(trow.x.iat[0]), float(trial.loc[trow.index,'z'].iat[0])
        #dist = math.hypot(dx, dz)
        distances.append(dx)

        # did we hit? (any 'h' in the slice)
        hit = 1 if ('h' in trial.EventType.values) else 0
        successes.append(hit)
        by_type.setdefault(ttype, []).append(hit)

    # 3) summarize
    if successes:
        overall = sum(successes)/len(successes)*100
        print(f"\n=== EVALUATION RESULTS ===")
        print(f"Overall success rate: {overall:.1f}% ({sum(successes)}/{len(successes)})")
        for ttype, hits in by_type.items():
            rate = sum(hits)/len(hits)*100
            print(f"  • Trial type {ttype}: {rate:.1f}% ({sum(hits)}/{len(hits)})")
    if distances:
        print(f"Max target distance reached: {max(distances):.3f}/5.00")
    print("==========================\n")

def train_solo(run_id, env_path, config_path, total_runs=5, log_name=None):
    next_run = get_next_run_number(run_id)
    run_id_list = []
    for i in range(total_runs):
        current = f"{run_id}_{next_run + i}"
        print(f"Starting training: {current}")
        # write the currentLog.txt
        fn = f"{(log_name if log_name else run_id)}_{next_run + i}_train.txt"       
        # sa = Path(env_path) / "2D go to target v1_Data" / "StreamingAssets" / "currentLog.txt"
        # sa.write_text(fn)
        
        sa = os.path.join(env_path,
                    "LinuxHeadless_Data",
                    "StreamingAssets",
                    "currentLog.txt")
    
        with open(sa, "w") as f:
            f.write(fn)

        time.sleep(1)
        cmd = [
            "mlagents-learn",
            config_path,
            "--env", str(Path(env_path) / "LinuxHeadless.x86_64"),
            "--run-id", current,
            "--force",
            "--env-args", "--screen-width=155", "--screen-height=86",
        ]
        subprocess.run(cmd, check=True)
        print(f"Completed training: {current}")
        time.sleep(5)
        run_id_list.append(current)
    return run_id_list

def train_multiple_networks(networks, env_path, runs_per_network=2, log_name=None, env='Normal'):
    run_id_list2 = []
    for network in networks:
        if network == "fully_connected":
            config_path = "./Config/fc.yaml"
        elif network == "simple":
            config_path = "./Config/simple.yaml"
        elif network == "resnet":
            config_path = "./Config/resnet.yaml"
        else:
            config_path = "./Config/nature.yaml"
            if network != "nature_cnn":
                replace.replace_nature_visual_encoder("C:/Users/mariu/Miniconda3/envs/mouse2/Lib/site-packages/mlagents/trainers/torch/encoders.py", "./Encoders/" + network + ".py")

        run_ids = train_solo(
            run_id=f"{network}_{env}",
            env_path=env_path,
            config_path=config_path,
            total_runs=runs_per_network,
            log_name=log_name
        )
        run_id_list2.extend(run_ids)
    return run_id_list2

# ─── New CLI entrypoint ──────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train multiple networks on MouseVsAI")
    parser.add_argument("--env", type=str, default="Normal",
                        help="Folder name under Builds/ to use as env")
    parser.add_argument("--runs-per-network", type=int, default=2,
                        help="How many runs per network variant")
    parser.add_argument("--networks", type=str, default="nature_cnn,simple,resnet",
                        help="Comma-separated list of network names")
    parser.add_argument("--log-name", type=str, default=None,
                        help="Optional prefix for all log files")
    args = parser.parse_args()

    env_folder = f"./Builds/{args.env}"
    nets = [n.strip() for n in args.networks.split(",")]
    run_ids = train_multiple_networks(nets, env_folder, args.runs_per_network, args.log_name, args.env)

    # Summarize each run
    logs_dir = "./logfiles"

    for rid in run_ids:
#    	if args.log_name:
#	    summary_filename = f"{args.log_name}_train.txt"
#	else:
#	    summary_filename = f"{rid}_train.txt"
	#summary_file = os.path.join(logs_dir, summary_filename)
        summary_file = os.path.join(logs_dir, f"{rid}_train.txt")

        print(f"\n=== Summary for {rid} ===")
        summarize_log(str(summary_file))
