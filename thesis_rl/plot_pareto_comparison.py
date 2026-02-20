import json
import matplotlib.pyplot as plt
import numpy as np


mappo_momarl = """
{
  "weights": [
    [0.0, 1.0],
    [0.3328, 0.6672],
    [0.6656, 0.3344],
    [1.0, 0.0]
  ],
  "vector_returns": [
    [ 97.5553, 364.5848],
    [133.551,  372.8791],
    [146.096,  317.0764],
    [153.1194, 335.5435]
  ],
  "objectives": ["inverse_global_temperature", "global_economic_output"]
}
"""

happo_momarl = """
{
  "weights": [
    [0.0, 1.0],
    [0.3328, 0.6672],
    [0.6656, 0.3344],
    [1.0, 0.0]
  ],
  "vector_returns": [
    [85.16510095596314, 358.18979360610246],
    [120.80234244465828, 358.4246584817767],
    [149.29237505793571, 342.6520297937095],
    [154.84534696936606, 289.1183898471296]
  ],
  "objectives": ["inverse_global_temperature", "global_economic_output"]
}
"""

def parse_data(json_str):
    data = json.loads(json_str)
    vector_returns = np.array(data["vector_returns"])
    objectives = data["objectives"]
    return vector_returns, objectives

def plot_pareto_front():
    # Parse data
    happo_returns, objectives = parse_data(happo_momarl)
    mappo_returns, _ = parse_data(mappo_momarl)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot HAPPO points
    ax.scatter(happo_returns[:, 0], happo_returns[:, 1], 
               c='#1f77b4', s=100, marker='o', label='HAPPO', alpha=0.8, edgecolors='black', linewidths=0.5)
    
    # Plot MAPPO points
    ax.scatter(mappo_returns[:, 0], mappo_returns[:, 1], 
               c='#ff7f0e', s=100, marker='s', label='MAPPO', alpha=0.8, edgecolors='black', linewidths=0.5)
    
    # Sort points by first objective for line plotting
    happo_sorted_idx = np.argsort(happo_returns[:, 0])
    mappo_sorted_idx = np.argsort(mappo_returns[:, 0])
    
    # Connect points with lines to show Pareto front
    ax.plot(happo_returns[happo_sorted_idx, 0], happo_returns[happo_sorted_idx, 1], 
            c='#1f77b4', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.plot(mappo_returns[mappo_sorted_idx, 0], mappo_returns[mappo_sorted_idx, 1], 
            c='#ff7f0e', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Format axis labels
    xlabel = objectives[0].replace('_', ' ').title()
    ylabel = objectives[1].replace('_', ' ').title()
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title('Pareto Front Comparison: HAPPO vs MAPPO', fontsize=14, fontweight='bold')
    
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('/home/olaand/JUSTICE/thesis_rl/pareto_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved plot to pareto_comparison.png and pareto_comparison.pdf")
    
    plt.show()

if __name__ == "__main__":
    plot_pareto_front()
