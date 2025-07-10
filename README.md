# Linux Training Guide

Welcome to **Mouse vs. AI: Robust Visual Foraging Challenge @ NeurIPS 2025**

This is a training guide for **Linux**. For other operating systems, please check:
[Windows](https://github.com/robustforaging/mouse_vs_ai_windows?tab=readme-ov-file#windows-training-guide) and [macOS](https://github.com/robustforaging/mouse_vs_ai_macOS?tab=readme-ov-file#macos-training-guide)

# Install conda
Open command prompt
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Press ```ENTER``` or type ```yes``` when prompted

# Create conda environment
Open command prompt and navigate to the directory where you want to download the project.

Clone the repository from GitHub:
```bash
git clone https://github.com/robustforaging/mouse_vs_ai_linux.git
cd mouse_vs_ai_linux
```

Then, create and activate the conda environment:
```bash
conda env create -n mouse -f mouse.yml
conda activate mouse
``` 
💡 Troubleshooting: if the CUDA version isn’t compatible with your GPU, please try: 
```bash
pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

# Set executable permissions for Linux binariesPermalink
To make the Linux executable files executable:
```bash
chmod +x ./Builds/RandomTrain/LinuxHeadless.x86_64

# If you have other build directories, make their executables executable too. For example:
# chmod +x ./Builds/RandomTest/LinuxHeadless.x86_64
```

# Modify file path
Open ```train.py``` and go to where ```replace.replace_nature_visual_encoder``` is called.
Update the path to point to the location of ```encoders.py``` in your conda environment.

💡 Tip: The ```encoders.py``` file is usually located in your conda environment’s working directory. For example: ```…/miniconda3/env/mouse2/Lib/site-packages/mlagents/trainers/torch/encoders.py```


# Run script


## Training
```text
Usage: python train.py [options]

Training options:
  --runs-per-network R    Number of runs per network (default: 5)
  --env ID                Run identifier (default: Normal) [defines type of environment]
  --network N1,N2,N3     Comma-separated list of networks to train
                         (default choices: ['fully_connected', 'nature_cnn', 'simple', 'resnet'])
```

Example command:

```bash
python train.py --runs-per-network 1 --env RandomTrain --network neurips,simple,fully_connected,resnet,alexnet
```
- Troubleshooting: If training only proceeds after pressing ```ENTER```, try running the command with unbuffered output mode:  ```python -u start.py train --runs-per-network 1 --env Normal --network neurips,simple,fully_connected``` 
- If the issue persists, stop the current training episode and train again

## Evaluating
Example command for evaluation:
```bash
python evaluate.py --model "/home/<your_username>/path/to/your_model.onnx" --log-name "example.txt" --episodes 10
```

# Customize the model
- To change architecture: Add your model to the `/mouse_vs_ai_linux/Encoders` folder
- To change hyperparamters: edit information in `/mouse_vs_ai_linux/Encoders/nature.yaml` file

Then run the above python training script.
