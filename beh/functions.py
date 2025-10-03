import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


def multi_split_violin_plot(
    data, hue_cols, y_col, n_rows, n_cols, labels_list=None, titles_list=None, violin_linewidth=0.5
):
    """
    Create multiple split violin plots with boxplot overlays in a grid.

    Parameters:
    data: DataFrame
    hue_cols: list of column names for grouping variables
    y_col: column name for y-axis variable
    n_rows, n_cols: grid dimensions
    labels_list: list of label dicts or None (default: {1: 'Tak', 2: 'Nie'})
    titles_list: list of titles or None (uses column names)
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 6 * n_rows))

    # Handle single plot case
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, hue_col in enumerate(hue_cols):
        if i >= len(axes):
            break

        # Filter data
        plot_data = data[data[hue_col].isin([1, 9])].copy()

        # Set labels
        if labels_list and i < len(labels_list) and labels_list[i]:
            labels = labels_list[i]
        else:
            labels = {1: "users", 9: "non-users"}

        plot_data["group_label"] = plot_data[hue_col].map(labels)

        # Set title
        if titles_list and i < len(titles_list):
            title = titles_list[i]
        else:
            title = hue_col
        # Create palette
        unique_labels = plot_data["group_label"].unique()

        palette_violin = {unique_labels[0]: "#07b5d4ff", unique_labels[1]: "#e287b9ff"}
        palette_boxplot = {unique_labels[0]: "#059bb6b7", unique_labels[1]: "#c578a1ad"}

        # Create violin plot without inner elements
        sns.violinplot(
            data=plot_data,
            x=[0] * len(plot_data),
            y=y_col,
            hue="group_label",
            split=True,
            inner=None,  # No inner elements
            fill=True,
            palette=palette_violin,
            density_norm="count",
            bw_adjust=0.4,
            linewidth=violin_linewidth,
            ax=axes[i],

        )

        # Add wider boxplot overlay
        sns.boxplot(
            data=plot_data,
            x=[0] * len(plot_data),
            y=y_col,
            hue="group_label",
            width=0.06,  # Control boxplot width
            ax=axes[i],
            showcaps=False,
            boxprops={'linewidth': 0.9},
            palette=palette_boxplot,
            whiskerprops={"linewidth": 0.9},
            medianprops={"linewidth": 1.3},
            showfliers=False,
        )

        # Add jittered data points split by group to match violin sides
        for label_idx, label in enumerate(sorted(unique_labels)):
            group_data = plot_data[plot_data["group_label"] == label]
            y_values = group_data[y_col].dropna()
            
            # Position points on correct side of violin split
            # First group (index 0) goes to right side, second group (index 1) goes to left side
            x_center = 0.06 if label_idx == 0 else -0.06
            # Create gaussian distribution around the center for each side
            x_positions = np.random.normal(x_center, 0.027, len(y_values))
            
            # Use darker colors for points
            point_color = "#034c56" if label == unique_labels[0] else "#7a4662"
            
            axes[i].scatter(
                x_positions,
                y_values,
                s=8,  # Small point size
                alpha=0.8,
                color=point_color,
                edgecolors='none',
                zorder=10  # High z-order to appear on top of boxplots
            )

        # Statistical testing
        if len(unique_labels) == 2:
            group1_data = plot_data[plot_data["group_label"] == unique_labels[0]][
                y_col
            ].dropna()
            group2_data = plot_data[plot_data["group_label"] == unique_labels[1]][
                y_col
            ].dropna()

            _, p_value = stats.ttest_ind(group1_data, group2_data)

        # Styling
        axes[i].set_title(title, fontweight="bold", fontsize=14)
        # Remove default xlabel - we'll add custom one below
        axes[i].set_xlabel("")
        # axes[i].set_ylabel(y_col if i == 0 else '', fontweight='bold')
        axes[i].set_ylabel("")

        # Set up y-axis with exact values
        y_min, y_max = axes[i].get_ylim()

        # Create nice tick marks - adjust the number of ticks as needed
        n_ticks = 6  # Number of ticks you want
        tick_values = np.linspace(y_min, y_max, n_ticks)
        axes[i].set_yticks(tick_values)

        # Format tick labels to show exact values (rounded to 1 decimal place)
        tick_labels = [f"{val:.1f}" for val in tick_values]
        axes[i].set_yticklabels(tick_labels, fontsize=10)

        # axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_xticks([])
        axes[i].legend().remove()

        # Remove spines
        for spine in axes[i].spines.values():
            spine.set_visible(False)

        # Add grid for better readability aligned with tick marks
        axes[i].grid(True, alpha=0.3, axis="y")
        axes[i].set_axisbelow(True)  # Put grid behind the plot elements

        # Add significance marker and line at the top of the plot
        if len(unique_labels) == 2:
            y_max_current = axes[i].get_ylim()[1]
            y_min_current = axes[i].get_ylim()[0]
            y_range = y_max_current - y_min_current

            # Position significance line and marker
            sig_line_y = y_max_current + y_range * 0.02
            sig_marker_y = y_max_current + y_range * 0.03

            # Draw horizontal line connecting the groups
            axes[i].plot([-0.15, 0.15], [sig_line_y, sig_line_y], "k-", linewidth=1)

            # Draw vertical lines down to the groups
            axes[i].plot(
                [-0.15, -0.15],
                [sig_line_y, sig_line_y - y_range * 0.01],
                "k-",
                linewidth=1,
            )
            axes[i].plot(
                [0.15, 0.15],
                [sig_line_y, sig_line_y - y_range * 0.01],
                "k-",
                linewidth=1,
            )

            # Add p-value (optional - small text)
            axes[i].text(
                0,
                sig_marker_y + y_range * 0.02,
                f"p={p_value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="gray",
            )

            # Extend y-axis to accommodate significance markers
            axes[i].set_ylim(y_min_current, y_max_current + y_range * 0.1)

        # Remove individual subplot frames
        for spine in axes[i].spines.values():
            spine.set_visible(False)

        # Add group labels and sample sizes below plot
        unique_labels_sorted = sorted(unique_labels)
        y_offset = (
            axes[i].get_ylim()[0]
            - (axes[i].get_ylim()[1] - axes[i].get_ylim()[0]) * 0.01
        )

        # Calculate sample sizes for each group
        group_counts = plot_data["group_label"].value_counts()

        # Add labels and sample sizes
        label1 = unique_labels_sorted[0]
        label2 = unique_labels_sorted[1]
        count1 = group_counts.get(label1, 0)
        count2 = group_counts.get(label2, 0)

        # Group labels
        axes[i].text(
            0.15,
            y_offset,
            label1,
            ha="center",
            transform=axes[i].transData,
            fontsize=12,
            fontweight="bold",
            color="black",
        )
        axes[i].text(
            -0.15,
            y_offset,
            label2,
            ha="center",
            transform=axes[i].transData,
            fontsize=12,
            fontweight="bold",
            color="black",
        )

        # Sample size labels below group labels
        y_offset_n = y_offset - (axes[i].get_ylim()[1] - axes[i].get_ylim()[0]) * 0.05
        axes[i].text(
            0.15,
            y_offset_n,
            f"n={count1}",
            ha="center",
            transform=axes[i].transData,
            fontsize=10,
            color="gray",
        )
        axes[i].text(
            -0.15,
            y_offset_n,
            f"n={count2}",
            ha="center",
            transform=axes[i].transData,
            fontsize=10,
            color="gray",
        )

        # # Add custom x-label below sample sizes (bold and centered)
        # y_offset_xlabel = (
        #     y_offset_n - (axes[i].get_ylim()[1] - axes[i].get_ylim()[0]) * 0.08
        # )
        # axes[i].text(
        #     0,
        #     y_offset_xlabel,
        #     "Psychedelics",
        #     ha="center",
        #     transform=axes[i].transData,
        #     fontsize=12,
        #     fontweight="bold",
        #     color="black",
        # )

    # Hide unused subplots
    for j in range(len(hue_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()