from functions import multi_split_violin_plot
import pandas as pd

data = pd.read_csv("beh_krk.csv")

# Short column names
short_colnames = [
    'new_mes',
    'City',
    'Group',
    'Gender',
    'Education',
    'Age',
    'Lifetime_psychedelics',
    'Lifetime_meditation_hours',
    'Alcohol',
    'Lifetime_cannabis',
    'RRQ_reflection',
    'RRQ_rumination',
    'BDI_II',
    'STAI_I',
    'STAI_II',
    'ARSQ_discontinuity_of_mind',
    'ARSQ_theory_of_mind',
    'ARSQ_self',
    'ARSQ_planning',
    'ARSQ_sleepiness',
    'ARSQ_comfort',
    'ARSQ_somatic_awareness',
    'ARSQ_health_concern',
    'ARSQ_visual_thought',
    'ARSQ_verbal_thought'
]

# Apply new column names
data.columns = short_colnames

int_cols = ['Age',
    'Lifetime_psychedelics',
    'Lifetime_meditation_hours',
    'Alcohol',
    'Lifetime_cannabis',
    'RRQ_reflection',
    'RRQ_rumination',
    'BDI_II',
    'STAI_I',
    'STAI_II',
    'ARSQ_discontinuity_of_mind',
    'ARSQ_theory_of_mind',
    'ARSQ_self',
    'ARSQ_planning',
    'ARSQ_sleepiness',
    'ARSQ_comfort',
    'ARSQ_somatic_awareness',
    'ARSQ_health_concern',
    'ARSQ_visual_thought',
    'ARSQ_verbal_thought'
   ]
# Convert to numeric then integer
for col in int_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce').astype('Float64')

# # Convert text columns to string
# data['psychedelic_other_motives'] = data['psychedelic_other_motives'].astype('string')


# Create ranked columns for frequency data
data['Lifetime_cannabis_rank'] = data['Lifetime_cannabis'].rank(method='dense')
data['Lifetime_meditation_hours_rank'] = data['Lifetime_meditation_hours'].rank(method='dense')

# multi_split_violin_plot(data,
#                        ['Group'],
#                        'BDI_II',
#                        1, 2,
#                     #    labels_list=[{1: 'Yes', 9: 'No'}],
#                        titles_list=['BDI-II'])

multi_split_violin_plot(data,
                       ['Group'],
                       'Lifetime_cannabis_rank',
                       1, 2,
                    #    labels_list=[{1: 'Yes', 9: 'No'}],
                       titles_list=['Lifetime Cannabis'])


# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np

# def multi_category_violin_plot(data, cat_cols, y_col, n_rows, n_cols,
#                               labels_list=None, titles_list=None):
#     """
#     Create multiple violin plots showing distribution of continuous variable across categorical answers.

#     Parameters:
#     data: DataFrame
#     cat_cols: list of categorical column names (e.g., 'mental_health_rating')
#     y_col: column name for continuous y-axis variable (e.g., '12_mo_freq_rank')
#     n_rows, n_cols: grid dimensions
#     labels_list: list of label dicts for each categorical variable (e.g., {1: 'Very Good', 2: 'Good', ...})
#     titles_list: list of titles or None (uses column names)
#     """
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 6*n_rows))

#     # Handle single plot case
#     if n_rows * n_cols == 1:
#         axes = [axes]
#     else:
#         axes = axes.flatten()

#     # Define a color palette for different categories
#     colors = ["#07b6d4", "#e287b9", "#4ade80", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#10b981"]

#     for i, cat_col in enumerate(cat_cols):
#         if i >= len(axes):
#             break

#         # Filter out missing values
#         plot_data = data.dropna(subset=[cat_col, y_col]).copy()

#       # Set labels for categories and maintain proper order
#         if labels_list and i < len(labels_list) and labels_list[i]:
#             labels = labels_list[i]
#             # Map the labels
#             plot_data['category_label'] = plot_data[cat_col].map(labels)
#             # Filter to only include mapped categories
#             plot_data = plot_data.dropna(subset=['category_label'])
#             # Get categories in the order of the original keys (1, 2, 3, 4, 5)
#             ordered_keys = sorted(labels.keys())
#             unique_categories = [labels[key] for key in ordered_keys if labels[key] in plot_data['category_label'].unique()]
#         else:
#             # Use original category values as labels, sorted by numeric value
#             plot_data['category_label'] = plot_data[cat_col].astype(str)
#             # Sort by the original numeric values
#             unique_values = sorted(plot_data[cat_col].unique())
#             unique_categories = [str(val) for val in unique_values]

#         # Set title
#         if titles_list and i < len(titles_list):
#             title = titles_list[i]
#         else:
#             title = cat_col

#         # Create color palette for this plot
#         palette = {cat: colors[j % len(colors)] for j, cat in enumerate(unique_categories)}

#      # Create violin plot with fixed category order
#         sns.violinplot(data=plot_data,
#                       x='category_label',
#                       y=y_col,
#                       order=unique_categories,  # This ensures proper ordering
#                       inner='box',  # Add box plot inside
#                       palette=palette,
#                       density_norm='count',
#                       bw_adjust=0.6,
#                       ax=axes[i])

#         # Styling
#         axes[i].set_title(title, fontweight='bold', fontsize=14)
#         axes[i].set_xlabel('')
#         # axes[i].set_ylabel(y_col if i == 0 else '', fontweight='bold')
#         axes[i].set_ylabel('Miesiące od ostatnich psychodelików')
#         axes[i].set_yticklabels([])
#         axes[i].tick_params(axis='x', rotation=45)
#         # axes[i].set_ylim(bottom=)

#         # Remove spines
#         for spine in axes[i].spines.values():
#             spine.set_visible(False)

#         # Add grid for better readability
#         axes[i].grid(True, alpha=0.3, axis='y')

#         # Add custom labels at center of y-axis using axes coordinates
#         axes[i].text(-0.04, 0.93, 'najwięcej', transform=axes[i].transAxes,
#                     ha='center', va='center', fontsize=8, fontweight='bold',
#                     rotation=90, color='gray')
#         axes[i].text(-0.04, 0.13, 'najmniej', transform=axes[i].transAxes,
#                     ha='center', va='center', fontsize=8, fontweight='bold',
#                     rotation=90, color='gray')

#         # Adjust x-axis labels if they're too long
#         labels = axes[i].get_xticklabels()
#         if any(len(label.get_text()) > 10 for label in labels):
#             axes[i].tick_params(axis='x', rotation=45)

#     # Hide unused subplots
#     for j in range(len(cat_cols), len(axes)):
#         axes[j].set_visible(False)

#     plt.tight_layout()
#     plt.show()

# # Example label dictionaries
# mental_health_labels = {
#     1: 'Bardzo dobre',
#     2: 'Dobre',
#     3: 'Średnie',
#     4: 'Złe',
#     5: 'Bardzo złe'
# }
# physical_health_labels = {
#     1: 'Bardzo dobre',
#     2: 'Dobre',
#     3: 'Średnie',
#     4: 'Złe',
#     5: 'Bardzo złe'
# }

# # Example usage with your data
# multi_category_violin_plot(
#     data=psychedelic_users,
#     cat_cols=['physical_health_rating', 'mental_health_rating'],
#     y_col='last_freq_rank',
#     n_rows=2,
#     n_cols=1,
#     labels_list=[physical_health_labels, mental_health_labels],
#     titles_list=['Samopoczucie fizyczne', 'Samopoczucie psychiczne']
# )

# # Alternative usage for single plot
# # multi_category_violin_plot(
# #     data=psychedelic_users,
# #     cat_cols=['mental_health_rating'],
# #     y_col='12_mo_freq_rank',
# #     n_rows=1,
# #     n_cols=1,
# #     labels_list=[mental_health_labels],
# #     titles_list=['Mental Health Rating vs Psychedelic Use Frequency']
# # )
