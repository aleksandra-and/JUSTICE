import json
import matplotlib.pyplot as plt
import numpy as np

# Data from temp_pareto.py
happo_momarl = """
{
  "weights": [
    [0.0, 1.0],
    [0.11037861872051274, 0.8896213812794873],
    [0.22165789720167345, 0.7783421027983266],
    [0.33297430052071014, 0.6670256994792899],
    [0.444311665511736, 0.555688334488264],
    [0.5557494120957494, 0.4442505879042506],
    [0.66700480091978, 0.33299519908022],
    [0.7783653189599546, 0.22163468104004522],
    [0.8896059662115817, 0.11039403378841832],
    [1.0, 0.0]
  ],
  "vector_returns": [
    [94.51551672518254, 222.67780303508044],
    [104.12224687337876, 227.59707519859074],
    [116.76099975705147, 232.19762840867043],
    [121.47486221194268, 233.5000894472003],
    [129.79510238170624, 235.61593861877918],
    [132.6323882460594, 236.29624560326337],
    [134.94277903437614, 236.748348338902],
    [131.95247903466225, 235.98048291355371],
    [131.19985050559043, 235.7258746340871],
    [129.49491734504699, 235.3038343027234]
  ],
  "objectives": ["inverse_global_temperature", "global_economic_output"]
}
"""

mappo_momarl = """
{
  "weights": [
    [0.0, 1.0],
    [0.11037861872051274, 0.8896213812794873],
    [0.22165789720167345, 0.7783421027983266],
    [0.33297430052071014, 0.6670256994792899],
    [0.444311665511736, 0.555688334488264],
    [0.5557494120957494, 0.4442505879042506],
    [0.66700480091978, 0.33299519908022],
    [0.7783653189599546, 0.22163468104004522],
    [0.8896059662115817, 0.11039403378841832],
    [1.0, 0.0]
  ],
  "vector_returns": [
    [98.27156089842319, 224.80501721054316],
    [105.70074980258941, 228.1312956213951],
    [110.80246379375458, 230.16082355231046],
    [119.92281673550606, 232.97114899903536],
    [123.97602691054344, 234.25951599925756],
    [118.78291693329811, 232.48907881975174],
    [125.1769595682621, 234.33392141312362],
    [124.90027307868004, 234.43726808577776],
    [118.734046626091, 232.22631165534258],
    [124.48728679418564, 234.0431743055582]
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
    plt.savefig('/home/olaand/JUSTICE/thesis_rl/pareto_comparison.pdf', bbox_inches='tight')
    print("Saved plot to pareto_comparison.png and pareto_comparison.pdf")
    
    plt.show()

if __name__ == "__main__":
    plot_pareto_front()
