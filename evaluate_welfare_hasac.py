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

def read_json_config(config_path):
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


if __name__ == "__main__":
    # HASAC Seeds
    load_config = "thesis_rl/configs/welfare/hasac_temp_welfare_prio.yaml"
    # "prioritarian", "utilitarian", 
    current_welfare_type = "prioritarian"

    #model_dir_base_path =  Path("results/welfare_hasac/hasac")
    model_dir_base_path =  Path("results/welfare_hasac/hasac")
    # Get all model directories that match the pattern "momarl_hasac_uniform_2026*_w*
    model_dirs = sorted(model_dir_base_path.glob("momarl_hasac_uniform_2026*_w*/seed-00042-2026-06-*/models"), key=lambda p: int(p.parent.parent.name.split('_')[4][1:]))

    evaluation_dir_base_path = Path(f"eval_results/welfare_analysis/{current_welfare_type}")
    evaluation_output_dirs = [evaluation_dir_base_path / f"hasac_seed_{model_dir.parent.name.split('-')[1]}_w{model_dir.parent.parent.name.split('_')[4][1:]}" for model_dir in model_dirs]

    for model_dir, evaluation_output_dir in zip(model_dirs, evaluation_output_dirs):
        seed = int(model_dir.parent.name.split('-')[1])
        # ex. results/welfare_hasac/hasac/momarl_hasac_uniform_202605_w9_9513/seed-00042-2026-06-01-00-33-25/config.yaml
        run_config_path = model_dir.parent / "config.json"
        welfare_type = read_json_config(run_config_path).get("env_args", {}).get("welfare_type", "utilitarian")
        if welfare_type != current_welfare_type:
            print(f"Skipping model in: \n  {model_dir} \n  because welfare type {welfare_type} does not match current welfare type {current_welfare_type}")
            continue
        print(f"Evaluating model in: \n  {model_dir} \n  and saving results to: \n  {evaluation_output_dir}")
        run_evaluation_script(load_config, model_dir, seed, evaluation_output_dir)
