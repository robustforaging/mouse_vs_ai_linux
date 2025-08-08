# evaluate.py

import argparse
import subprocess
from pathlib import Path
from train import summarize_log
import os

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate an ONNX model on MouseVsAI")
    p.add_argument("--model", required=True,
                   help="Path to the trained ONNX model file")
    p.add_argument("--episodes", type=int, default=50,
                   help="Number of episodes to run in inference")
    p.add_argument("--env", type=str, default="RandomTest",
                   help="Build folder name under ./Builds/")
    p.add_argument("--log-name", required=True,
                   help="Base name for the output log file")
    return p.parse_args()

def main():
    args = parse_args()

    env_path = os.path.join("Builds", args.env)

    exe = os.path.join(env_path,"LinuxHeadless.x86_64")
    
    # Prepare the Unity log filename

    logs_dir = "./logfiles"
    base_name = f"{args.log_name}_test"
    ext = ".txt"
    log_fn = base_name + ext
    summary_path = os.path.join(logs_dir,log_fn)

    counter = 1
    while os.path.exists(summary_path):
        log_fn = f"{base_name}_{counter}{ext}"
        counter += 1
        # Now summarize
        summary_path = os.path.join(logs_dir,log_fn)


    sa = os.path.join(env_path,
                    "LinuxHeadless_Data",
                    "StreamingAssets",
                    "currentLog.txt")
    
    with open(sa, "w") as f:
        f.write(log_fn)
    # Run in inference-only mode
    cmd = [
        str(exe),
        f"--model={args.model}",
        f"--episodes={args.episodes}",
        # you can also add "-batchmode", "-nographics" here if desired
    ]
    print("[EVAL] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    print(f"\n=== Evaluation Summary ({log_fn}) ===")
    summarize_log(str(summary_path))

if __name__ == "__main__":
    main()
