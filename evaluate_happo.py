from pathlib import Path
import shlex
import subprocess

def run_evaluation_script(config_path, model_dir, seed, evaluation_output_dir):
    config_path = Path(config_path)
    model_dir = Path(model_dir)
    evaluation_output_dir = Path(evaluation_output_dir)
    evaluation_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "./thesis_rl/evaluation.py",
        "--load_config", str(config_path),
        "--seed", str(seed),
        "--model_dir", str(model_dir),
        "--evaluation_output_dir", str(evaluation_output_dir),
    ]

    # Switch to root directory of the project
    project_root = Path.cwd().resolve()
    subprocess.run(cmd, check=True, cwd=project_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# HAPPO Seeds
#project_root = Path.cwd().resolve().parent.parent
load_config = "thesis_rl/configs/algo_comparison/happo_temp_econ_continous_v2.yaml"

model_dir_base_path =  Path("results/continous_3_fixed/harl_justice_momarl/harl_justice_momarl/happo")
# Get all model directories that match the pattern "momarl_happo_uniform_2026*_w*
model_dirs = sorted(model_dir_base_path.glob("momarl_happo_uniform_2026*_w*/seed-0000*-*/models"), key=lambda p: int(p.parent.parent.name.split('_')[4][1:]))

evaluation_dir_base_path = Path("eval_results/detailed_analysis/evaluation/happo")
evaluation_output_dirs = [evaluation_dir_base_path / f"happo_seed_{model_dir.parent.name.split('-')[1]}_w{model_dir.parent.parent.name.split('_')[4][1:]}" for model_dir in model_dirs]

for model_dir, evaluation_output_dir in zip(model_dirs, evaluation_output_dirs):
    seed = int(model_dir.parent.name.split('-')[1])
    print(f"Evaluating model in: \n  {model_dir} \n  and saving results to: \n  {evaluation_output_dir}")
    run_evaluation_script(load_config, model_dir, seed, evaluation_output_dir)