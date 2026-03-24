# export PYTHONPATH=$PYTHONPATH:/Users/palokbiswas/Desktop/pollockdevis_git/JUSTICE/
from justice.util.output_data_processor import process_scenario

import os
import sys
import filecmp
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from functools import partial
from justice.util.enumerations import WelfareFunction, SSP
from justice.util.output_data_processor import (
    process_scenario,
    generate_reference_set_policy_mapping,
    read_reference_set_policy_mapping,
)

from justice.util.output_data_processor import (
    reevaluate_optimal_policy,
    reevaluated_optimal_policy_variable_extractor,
)
from justice.util.model_time import TimeHorizon
from justice.util.data_loader import DataLoader


def process_scenario_parallel(
    start_year,
    end_year,
    data_timestep,
    timestep,
    scenario_list,
    social_welfare_function,
    ssp,
    base_path,
):
    data_loader = DataLoader()
    region_list = data_loader.REGION_LIST

    time_horizon = TimeHorizon(
        start_year=start_year,
        end_year=end_year,
        data_timestep=data_timestep,
        timestep=timestep,
    )
    list_of_years = time_horizon.model_time_horizon

    sw_name = social_welfare_function.value[1]
    ssp_name = ssp.name  # str(ssp).split(".")[1]
    path = base_path + sw_name + "_" + ssp_name + "/"
    # path = os.path.join(base_path, ssp_name, "")
    filename = f"{sw_name}_reference_set.csv"
    csv_path = os.path.join(path, filename)

    loaded_df = pd.read_csv(csv_path)
    policy_indices = list(range(len(loaded_df)))

    print(f"Loading data for {sw_name} from {csv_path}")
    print("Selected policy-indices last 2 columns:")
    print(loaded_df.iloc[policy_indices, -2:])

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    bound_process_scenario = partial(
        process_scenario, social_welfare_function, path, policy_indices
    )
    with mp.Pool(processes=len(scenario_list)) as pool:
        pool.map(bound_process_scenario, scenario_list)


if __name__ == "__main__":

    # Get  swf, ssp, base_dir from sys.argv or set default values
    base_dir = sys.argv[1] if len(sys.argv) > 4 else "data/temporary/NU_DATA/mmBorg/"
    swf_input = (sys.argv[2] if len(sys.argv) > 2 else "PRIORITARIAN").upper()
    ssp_name = (sys.argv[3] if len(sys.argv) > 3 else "SSP1").upper()

    ssp = SSP[ssp_name]

    social_welfare_function = WelfareFunction[swf_input]

    scenario_list = ["SSP126", "SSP245", "SSP370", "SSP460", "SSP534"]

    print(f"Selected Social Welfare Function: {social_welfare_function}, SSP: {ssp}")

    ########################################

    # Step 1: Parallel Scenario Processing Block which produces welfare and temperature data for all policies and scenarios and ensemble members
    process_scenario_parallel(
        start_year=2015,
        end_year=2300,
        data_timestep=5,
        timestep=1,
        scenario_list=scenario_list,
        social_welfare_function=social_welfare_function,
        ssp=ssp,
        base_path=base_dir,
    )
    ########################################
    # Step 2: Generate Mapping from policy index to objective values
    mapping = generate_reference_set_policy_mapping(
        swf=social_welfare_function,
        data_root=Path(base_dir + social_welfare_function.value[1] + "_" + ssp.name),
        scenario_list=scenario_list,
        saving=True,
        output_directory="mapping",
        delete_loaded_files=True,  # Set to True to delete the loaded files after processing
    )
    ########################################
