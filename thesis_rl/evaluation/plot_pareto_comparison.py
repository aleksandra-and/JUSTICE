import json
import matplotlib.pyplot as plt
import numpy as np


mappo_momarl = """
{
  "weights": [
    [
      0.0,
      1.0
    ],
    [
      0.11037861872051274,
      0.8896213812794873
    ],
    [
      0.22165789720167345,
      0.7783421027983266
    ],
    [
      0.33297430052071014,
      0.6670256994792899
    ],
    [
      0.444311665511736,
      0.555688334488264
    ],
    [
      0.5557494120957494,
      0.4442505879042506
    ]
  ],
  "vector_returns": [
    [
      87.22878960072994,
      361.40145359933376
    ],
    [
      96.93779152929783,
      353.0493791565299
    ],
    [
      115.59008805155754,
      353.1630643621087
    ],
    [
      128.68684587478637,
      352.50225507318976
    ],
    [
      143.6835015773773,
      347.0631626471877
    ],
    [
      146.4962388753891,
      369.44177739322186
    ]
  ],
  "objectives": [
    "inverse_global_temperature",
    "global_economic_output"
  ]
}
"""

happo_momarl = """
{
  "weights": [
    [
      0.0,
      1.0
    ],
    [
      0.11037861872051274,
      0.8896213812794873
    ],
    [
      0.22165789720167345,
      0.7783421027983266
    ],
    [
      0.33297430052071014,
      0.6670256994792899
    ],
    [
      0.444311665511736,
      0.555688334488264
    ],
    [
      0.5557494120957494,
      0.4442505879042506
    ],
    [
      0.66700480091978,
      0.33299519908022
    ],
    [
      0.7783653189599546,
      0.22163468104004522
    ],
    [
      0.8896059662115817,
      0.11039403378841832
    ],
    [
      1.0,
      0.0
    ]
  ],
  "vector_returns": [
    [
      80.2478672683239,
      352.6750976368785
    ],
    [
      95.41571840047837,
      372.3851905450225
    ],
    [
      117.18392018079757,
      371.08393197655676
    ],
    [
      135.29135771989823,
      388.125299552083
    ],
    [
      141.14919379353523,
      386.2724945276976
    ],
    [
      151.04843289256095,
      368.1447522237897
    ],
    [
      151.63313980102538,
      341.04154691919683
    ],
    [
      155.536719340086,
      305.3578864239156
    ],
    [
      155.87657739520074,
      309.772470767051
    ],
    [
      158.95576633810998,
      266.0268811404705
    ]
  ],
  "objectives": [
    "inverse_global_temperature",
    "global_economic_output"
  ]
}
"""

def is_pareto_efficient(points):
    """Return a boolean mask of Pareto-efficient points (maximization)."""
    n_points = points.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                # Check if point j dominates point i (all >= and at least one >)
                if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                    is_efficient[i] = False
                    break
    return is_efficient

def parse_data(json_str):
    data = json.loads(json_str)
    vector_returns = np.array(data["vector_returns"])
    weights = np.array(data["weights"])
    objectives = data["objectives"]
    
    # Filter to Pareto front
    pareto_mask = is_pareto_efficient(vector_returns)
    vector_returns = vector_returns[pareto_mask]
    weights = weights[pareto_mask]
    
    return vector_returns, weights, objectives

def plot_pareto_front():
    # Parse data
    happo_returns, happo_weights, objectives = parse_data(happo_momarl)
    mappo_returns, mappo_weights, _ = parse_data(mappo_momarl)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot HAPPO points
    ax.scatter(happo_returns[:, 0], happo_returns[:, 1], 
               c='#1f77b4', s=100, marker='o', label='HAPPO', alpha=0.8, edgecolors='black', linewidths=0.5)
    
    # Plot MAPPO points
    ax.scatter(mappo_returns[:, 0], mappo_returns[:, 1], 
               c='#ff7f0e', s=100, marker='s', label='MAPPO', alpha=0.8, edgecolors='black', linewidths=0.5)
    
    # Add weight labels for HAPPO points
    for i, (x, y) in enumerate(happo_returns):
        weight_str = f"({happo_weights[i, 0]:.2f}, {happo_weights[i, 1]:.2f})"
        ax.annotate(weight_str, (x, y), textcoords="offset points", xytext=(5, 5), 
                    fontsize=8, color='#1f77b4')
    
    # Add weight labels for MAPPO points
    for i, (x, y) in enumerate(mappo_returns):
        weight_str = f"({mappo_weights[i, 0]:.2f}, {mappo_weights[i, 1]:.2f})"
        ax.annotate(weight_str, (x, y), textcoords="offset points", xytext=(5, -10), 
                    fontsize=8, color='#ff7f0e')
    
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
