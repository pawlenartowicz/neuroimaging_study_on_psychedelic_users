import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

def plot_specification_curve(spec_df, results_df, output_file=None, title="Specification Curve Analysis", 
                            figsize=(12, None), plot_type='barcode'):
    """
    Creates a specification curve visualization with either barcode-like display or color density
    for conditions, depending on the number of specifications.
    
    Parameters:
        spec_df (DataFrame): DataFrame containing specification factors
        results_df (DataFrame): DataFrame containing contrast results
        output_file (str, optional): Path to save the plot. If None, displays the plot.
        title (str, optional): Title for the plot.
        figsize (tuple): Figure width and height. If height is None, it will be calculated
                         based on the number of conditions.
        plot_type (str): 'barcode' forces traditional barcode visualization
                         'density' forces density-based visualization
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    
    # Skip visualization if no data
    if len(spec_df) == 0:
        print("No data available for specification curve analysis.")
        return None
    
    # Prepare for visualization
    # Combine with results for sorting
    combined_df = pd.concat([spec_df, results_df[["contrast"]]], axis=1)
    combined_df = combined_df.sort_values("contrast")
    
    # Extract condition columns (excluding node_pair)
    condition_columns = [col for col in spec_df.columns if col != "node_pair"]
    
    # Calculate total number of rows needed and row positions
    row_positions = {}
    current_pos = 0
    
    for col in condition_columns:
        unique_values = sorted(combined_df[col].dropna().unique())
        
        # Store row positions
        row_positions[col] = {}
        row_positions[col]['header'] = current_pos  # Position for condition name
        current_pos += 0.8  # Space after header
        
        # Set positions for values
        for i, val in enumerate(unique_values):
            row_positions[col][val] = current_pos + i
        
        current_pos += len(unique_values) + 0.5  # Space after group
    
    total_rows = current_pos
    
    # Calculate figure height if not provided
    if figsize[1] is None:
        fig_height = 5 + (total_rows * 0.25)
    else:
        fig_height = figsize[1]
    
    # Create the figure and axes explicitly
    fig = plt.figure(figsize=(figsize[0], fig_height))
    
    # Create axes with specific height ratios (make top panel smaller)
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=1)  # Top 1/6 for scatter plot
    ax2 = plt.subplot2grid((6, 1), (1, 0), rowspan=5)  # Bottom 5/6 for barcode/density
    
    # Plot the effect size (contrast) in top panel
    x_vals = np.arange(len(combined_df))
    y_vals = combined_df["contrast"].values
    
    # Create scatter plot with color gradient
    points = ax1.scatter(x_vals, y_vals, alpha=0.7, s=30, c=y_vals, cmap="viridis")
    ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax1.set_ylabel("Contrast Effect")
    ax1.set_title(title)
    ax1.set_xticks([])
    
    # Calculate median and mean contrast
    median_contrast = np.median(y_vals)
    mean_contrast = np.mean(y_vals)
    
    # Add stats box in the lower right corner, but within plot borders
    stats_text = f"Mean: {mean_contrast:.3f}\nMedian: {median_contrast:.3f}"
    
    # Get axis limits to position the box properly
    y_min, y_max = ax1.get_ylim()
    x_min, x_max = ax1.get_xlim()
    
    # Position the text box more to the left (90% instead of 98% of x_max)
    # This ensures it stays within the plot borders
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax1.text(x_max * 0.90, y_min + (y_max - y_min) * 0.15, stats_text, 
             ha='right', va='bottom', bbox=props, fontsize=9)
    
    # Reduce margins around data points in scatter plot
    ax1.margins(x=0.01, y=0.1)  # Much smaller margins
    
    # Set up bottom panel for barcode display
    ax2.set_xlim(0, len(combined_df))
    ax2.set_ylim(total_rows + 0.5, -0.5)  # Flip y-axis
    ax2.set_xlabel("Specification Number (sorted by effect size)")
    ax2.set_yticks([])
    ax2.grid(False)
    
    # Position for labels on the left
    label_offset = -0.01 * len(combined_df)
    
    # Determine which plot type to use
    num_specs = len(combined_df)
    if plot_type == 'barcode':
        use_density = False
        if num_specs > 2000:
            print("High number of specifications, consider using plot_type = 'density' for better readability, but beware of longer visualization time")
    elif plot_type == 'density':
        use_density = True
    
    # Get the coolwarm colormap for blue-to-red gradient
    cmap = plt.cm.coolwarm
    
    # Calculate the total number of data points
    total_data_points = len(combined_df)
    
    # Create the appropriate visualization for each condition
    for col in condition_columns:
        unique_values = sorted(combined_df[col].dropna().unique())
        
        # Add condition name as header
        header_row = row_positions[col]['header']
        ax2.text(label_offset, header_row, col, ha='right', va='center', 
                fontweight='bold', fontsize=11)
        
        # Process each value
        for val in unique_values:
            row = row_positions[col][val]
            
            # Plot value label
            ax2.text(label_offset, row, str(val), ha='right', va='center', fontsize=9)
            
            # Find positions where this value occurs
            mask = combined_df[col] == val
            positions = x_vals[mask]
            
            if len(positions) > 0:
                if use_density:
                    # For density plot, create a filled area with color based on density
                    # First create a density array
                    density = np.zeros(len(combined_df))
                    density[positions] = 1
                    
                    # Smooth the density for better visualization
                    window_size = max(3, int(num_specs / 500))
                    if window_size % 2 == 0:
                        window_size += 1  # Ensure odd window size
                    
                    smoothed = gaussian_filter1d(density, sigma=window_size/6)
                    
                    # Calculate the actual proportion of this value in the dataset
                    actual_proportion = len(positions) / total_data_points
                    
                    # Normalize density relative to expected proportion
                    # Values close to 1.0 indicate the density is as expected
                    # Values above 1.0 indicate higher than expected density
                    # Values below 1.0 indicate lower than expected density
                    normalized = np.zeros_like(smoothed)
                    if actual_proportion > 0:
                        normalized = smoothed / actual_proportion
                    
                    # Create colored density visualization
                    for i in range(len(smoothed)):
                        if smoothed[i] > 0.01:  # Only plot visible density
                            # Get color from colormap (0.5 is neutral)
                            # 0.0 (blue): much less than expected based on proportion
                            # 0.5 (purple): as expected based on proportion
                            # 1.0 (red): much more than expected based on proportion
                            color_value = min(1.0, normalized[i] / 2.0)
                            color = cmap(color_value)
                            
                            # Plot as a filled rectangle with color based on density
                            rect = plt.Rectangle((i-0.4, row-0.4), 0.8, 0.8, 
                                              facecolor=color, edgecolor='none', alpha=0.8)
                            ax2.add_patch(rect)
                else:
                    # For barcode, use vertical lines
                    ax2.vlines(positions, row - 0.4, row + 0.4, colors='k', linewidth=0.5)
    
    # Adjust layout with very tight spacing
    plt.subplots_adjust(left=0.18, right=0.98, top=0.95, bottom=0.05, hspace=0.01)
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    return fig

# For future
def visualize_force_directed_graph(nodes_list, contrasts_list, node_labels=None):
    """
    Creates a force-directed graph visualization using NetworkX and Matplotlib.
    Uses node_labels to color nodes by network if available.
    
    Parameters:
    nodes_list: List of node pairs (tuples)
    contrasts_list: List of contrast values corresponding to node pairs
    node_labels: Dictionary mapping nodes to their network labels (e.g., 'DMN', 'SN', 'CEN')
    
    Returns:
    None (displays the graph)
    """
    # Create a graph
    G = nx.Graph()
    
    # Add edges with contrast values as weights
    for (node_pair, contrast) in zip(nodes_list, contrasts_list):
        # Add edge with weight attribute
        G.add_edge(node_pair[0], node_pair[1], weight=contrast)
    
    # Create positions using force-directed layout
    # k is the optimal distance between nodes
    # Higher weights = stronger attraction
    pos = nx.spring_layout(G, k=0.15, iterations=100, 
                           weight='weight', seed=42)
    
    # Get edge weights for line thickness
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Normalize weights for visualization (larger contrast = thicker line)
    max_weight = max(edge_weights) if edge_weights else 1
    normalized_weights = [3 * w / max_weight for w in edge_weights]
    
    # Draw the graph
    plt.figure(figsize=(10, 8))
    
    # Set up colors for networks
    network_colors = {
        'DMN': 'royalblue',
        'SN': 'crimson',
        'CEN': 'forestgreen',
        None: 'gray'  # Default color for unlabeled nodes
    }
    
    # Assign colors to nodes based on their network
    if node_labels:
        node_colors = [network_colors.get(node_labels.get(node, None), 'gray') for node in G.nodes()]
        
        # Add legend for networks
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=color, markersize=10, label=network)
                           for network, color in network_colors.items() if network]
        plt.legend(handles=legend_elements, loc='upper right')
    else:
        # Default color if no labels
        node_colors = ['skyblue' for _ in G.nodes()]
    
    # Draw edges with varying thickness based on weight
    nx.draw_networkx_edges(G, pos, width=normalized_weights, 
                          alpha=0.7, edge_color='gray')
    
    # Draw nodes with colors based on network
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, 
                          alpha=0.8, edgecolors='black')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
    
    # Draw edge labels (contrast values) with reduced visibility
    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                font_size=8,
                                font_color='gray',
                                alpha=0.6,
                                bbox=dict(boxstyle="round,pad=0.1", 
                                         alpha=0.2,
                                         ec="none",
                                         fc="white"))
    
    plt.title("Force-Directed Graph with Contrast Values", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    plt.show()