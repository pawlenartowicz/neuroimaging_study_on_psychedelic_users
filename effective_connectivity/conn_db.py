# Standard library imports
import os
import pickle
import gzip
from copy import deepcopy

# Third-party imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.formula.api as smf

# Local imports
from helpers_functions import *
from visualizations import plot_specification_curve

class Study:
    def __init__(self, nodes=None, control_group_name=None):
        if nodes is None: 
            self.node_list = set()
            self.node_labels = None
        elif isinstance(nodes, dict):
            self.node_list = set(nodes.keys())
            self.node_labels = nodes
        else: 
            raise ValueError("Error importing nodes")
        
        self.data = {}
        self.control_group_name = control_group_name  # Store the control group name
        self.samples = {}  # Store sample information
    
    def _generate_sample_id(self, measurement_conditions, independent_samples):
        """
        Generate a sample ID based on the independent sample conditions.
        
        Parameters:
            measurement_conditions: Dictionary of measurement conditions
            independent_samples: List of condition keys that indicate independent samples
            
        Returns:
            A string representing the sample ID
        """
        if not independent_samples or not measurement_conditions:
            return "default_sample"
        
        # Generate sample ID from independent sample conditions
        sample_parts = []
        for condition in independent_samples:
            if condition in measurement_conditions:
                sample_parts.append(f"{condition}_{measurement_conditions[condition]}")
        
        if not sample_parts:
            return "default_sample"
        
        return "_".join(sample_parts)

    def import_measurement_from_excel(self, dest,  
                                    name=None,
                                    measurement_conditions=None,
                                    skip_col=[1], id_col=2, condition_col=3, 
                                    node_names={"sep":"_","skip":1},
                                    calculate_contrasts=True,
                                    independent_samples=[]):
        """
        Import measurement data from Excel file with precomputed ranks for faster permutation testing.
        
        Parameters:
            dest: Path to Excel file
            name: Name for the measurement (if None, extracted from filename)
            measurement_conditions: Dictionary of measurement conditions
            skip_col: List of column indices to skip
            id_col: Column index containing subject IDs
            condition_col: Column index containing group assignments
            node_names: Parameters for extracting node names from column headers
            calculate_contrasts: Whether to calculate contrast effects during import
            independent_samples: List of condition keys that indicate independent samples
        """
        if name is None:
            name = extract_filename(dest)
        
        # Extract data - read only once
        excel_data = pd.read_excel(dest)
        
        # Pre-extract id and condition columns to avoid repeated list conversion
        id_list = excel_data.iloc[:, id_col-1].tolist()
        id_array = np.array(id_list)
        group = excel_data.iloc[:, condition_col-1].tolist()
        group_array = np.array(group)
        
        # Generate sample ID based on independent samples
        sample_id = self._generate_sample_id(measurement_conditions, independent_samples)
        
        sample_data = {
            'ID': id_array,         # Keep as original type array
            'group': group_array    # Keep as original type array
        }
        new_sample_df = pd.DataFrame(sample_data)
        
        # Check if this sample already exists and update it
        if sample_id in self.samples:
            existing_df = self.samples[sample_id]
            
            # For each subject in the new data, check if they exist in the current sample
            for i, subject_id in enumerate(id_array):
                # Check if this subject already exists
                existing_subject = existing_df[existing_df['ID'] == subject_id]
                
                if not existing_subject.empty:
                    # Check if group assignment is consistent
                    existing_group = existing_subject['group'].iloc[0]
                    current_group = group_array[i]
                    
                    if existing_group != current_group:
                        # Raise an error instead of just printing a message
                        raise ValueError(
                            f"Subject {subject_id} has inconsistent group assignments: "
                            f"existing={existing_group}, current={current_group} in file {dest}"
                        )
            
            # Concatenate with new data and remove duplicates
            self.samples[sample_id] = pd.concat([existing_df, new_sample_df]).drop_duplicates(subset=['ID']).reset_index(drop=True)
        else:
            # Create a new sample DataFrame
            self.samples[sample_id] = new_sample_df
        
        # Process data columns in bulk
        for i, column_name in enumerate(excel_data.columns):
            col_idx = i + 1  # Convert to 1-based indexing
            
            if col_idx in skip_col or col_idx == id_col or col_idx == condition_col:
                continue  # Skip these columns
                
            # Extract column values once
            column_values = excel_data.iloc[:, i].tolist()
            values_array = np.array(column_values, dtype=float)
            
            # Process the data column
            nodes = extract_colname(column_name, node_names)
            self.node_list.update(nodes)
            
            new_name = f"{name}_{nodes[0]}_{nodes[1]}"
            
            # Precompute ranks if requested
            ranks = None
            if calculate_contrasts:
                # Filter out NaN values
                valid_mask = ~np.isnan(values_array)
                
                if np.any(valid_mask):
                    # Get valid values and corresponding groups
                    valid_values = values_array[valid_mask]
                    
                    # Calculate ranks for all valid values
                    sorted_indices = np.argsort(valid_values)
                    temp_ranks = np.zeros(len(valid_values))
                    temp_ranks[sorted_indices] = np.arange(1, len(valid_values) + 1)
                    
                    # Handle ties
                    unique_values, counts = np.unique(valid_values, return_counts=True)
                    for j, count in enumerate(counts):
                        if count > 1:  # It's a tie
                            value = unique_values[j]
                            tie_mask = valid_values == value
                            temp_ranks[tie_mask] = np.mean(temp_ranks[tie_mask])
                    
                    # Create a full array with NaNs in invalid positions
                    ranks = np.full_like(values_array, np.nan)
                    ranks[valid_mask] = temp_ranks

                between_group_contrast = contrast_effect(ranks, group_array, self.control_group_name)
            else:
                between_group_contrast = None
                
            self.data[new_name] = {   
                "measurement_conditions": measurement_conditions,
                "nodes": nodes,
                "id": id_list,
                "sample_id": sample_id,  # Store sample_id instead of group
                "values": column_values,
                "ranks": ranks,
                "between_group_contrast": between_group_contrast
            }

    def specification_data(self, target_nodes=None, add_network_categories=False, filter=None):
        """
        Collects and prepares data for specification curve analysis.
        
        Parameters:
            target_nodes (tuple, optional): Specific node pair to analyze. If None, analyzes all node pairs.
            add_network_categories (bool): Whether to add network relationship categories.
            filter (tuple, optional): A tuple of ("include", dict) or ("exclude", dict) to filter data.
                                dict can contain single values or lists of values for each key.
                                If None, no additional filtering is applied.
        
        Returns:
            tuple: (specifications_df, results_df) containing the analysis data
        """
        
        # Prepare data structures for the analysis
        specifications = []
        results = []
        
        # Check if we can add network categories
        can_add_networks = add_network_categories and self.node_labels is not None
        
        # Process each measurement
        for name, item in self.data.items():
            # Skip if not containing contrast data
            if "between_group_contrast" not in item or item["between_group_contrast"] is None:
                continue
            
            if not isinstance(item["between_group_contrast"], (int, float)):
                continue
            
            # Skip if not the target nodes (if specified)
            if target_nodes is not None and tuple(item["nodes"]) != target_nodes:
                continue
            
            # Skip if doesn't pass the filter
            if not passes_filter(item.get("measurement_conditions"), filter):
                continue
            
            # Extract nodes and contrast
            node_pair = tuple(item["nodes"])
            contrast = item["between_group_contrast"]
            
            # Extract measurement conditions
            spec_dict = {"node_pair": f"{node_pair[0]}→{node_pair[1]}"}
            
            # Add network relationship category if possible
            if can_add_networks:
                source_node, target_node = node_pair
                if source_node in self.node_labels and target_node in self.node_labels:
                    source_network = self.node_labels[source_node] # type: ignore
                    target_network = self.node_labels[target_node] # type: ignore
                    spec_dict["network_relation"] = f"{source_network}→{target_network}"
            
            if item.get("measurement_conditions"):
                for key, value in item["measurement_conditions"].items():
                    spec_dict[key] = value
            
            # Add data to our lists
            specifications.append(spec_dict)
            results.append({
                "contrast": contrast,
                "name": name,
                "node_pair": f"{node_pair[0]}→{node_pair[1]}"
            })
        
        # Convert to DataFrames
        spec_df = pd.DataFrame(specifications)
        results_df = pd.DataFrame(results)
        
        return spec_df, results_df

    def specification_curve(self, target_nodes=None, output_file=None, title="Specification Curve Analysis", 
                        sample_size=None, random_seed=None, add_network_categories=False,
                        plot_type='barcode', filter=None):
        """
        Performs specification curve analysis on the contrasts, using measurement conditions
        as specification factors.
        
        Parameters:
            target_nodes (tuple, optional): Specific node pair to analyze. If None, analyzes all node pairs.
            output_file (str, optional): Path to save the plot. If None, displays the plot.
            title (str, optional): Title for the plot.
            sample_size (int, optional): If provided, randomly samples this many data points for faster testing.
            random_seed (int, optional): Seed for reproducible random sampling.
            add_network_categories (bool): Whether to add network relationship categories (DMN→DMN, CEN→SN, etc).
            plot_type (str): 'barcode', or 'density' - determines visualization method
                            'barcode' forces traditional barcode visualization
                            'density' forces density-based visualization
            filter (tuple, optional): A tuple of ("include", dict) or ("exclude", dict) to filter data.
                                dict can contain single values or lists of values for each key.
                                If None, no additional filtering is applied.
            
        Returns:
            tuple: (specifications_df, results_df) containing the analysis data
        """
        
        # Get the data with optional network categorization and filtering
        spec_df, results_df = self.specification_data(
            target_nodes=target_nodes, 
            add_network_categories=add_network_categories,
            filter=filter
        )
        
        # Sample data if requested
        if sample_size is not None and len(spec_df) > sample_size:
            
            # Set random seed if provided
            if random_seed is not None:
                np.random.seed(random_seed)
            
            # Sample indices
            sample_indices = np.random.choice(len(spec_df), size=sample_size, replace=False)
            
            # Filter DataFrames
            spec_df = spec_df.iloc[sample_indices].reset_index(drop=True)
            results_df = results_df.iloc[sample_indices].reset_index(drop=True)
            
            # Update title to indicate sampling
            title = f"{title} (Sample: {sample_size} points)"
        
        # Create the visualization
        if len(spec_df) > 0:
            plot_specification_curve(
                spec_df, 
                results_df, 
                output_file, 
                title,
                plot_type=plot_type
            )
        else:
            print("No data available for specification curve analysis.")
        
        return spec_df, results_df
    
    def permute(self, n_permutations=1000, column="group", seed=None):
        if seed is not None:
            np.random.seed(seed)
                
        # Filter valid measurements
        valid_measurements = {k: v for k, v in self.data.items()
                            if "between_group_contrast" in v and v["between_group_contrast"] is not None
                            and "ranks" in v and v["ranks"] is not None
                            and "sample_id" in v}
        
        keys = list(valid_measurements.keys())
        
        # Pre-process samples
        sample_lookup = {}
        for sample_id, sample_df in self.samples.items():
            if column in sample_df.columns:
                sample_lookup[sample_id] = {
                    'ids': sample_df['ID'].values,
                    'values': sample_df[column].values
                }
        
        # Cache for valid indices and IDs - do this work ONCE
        measurement_cache = {}
        for key in keys:
            measurement = valid_measurements[key]
            ranks = measurement["ranks"]
            ids = measurement["id"]
            sample_id = measurement["sample_id"]
            
            # Skip if sample not in lookup
            if sample_id not in sample_lookup:
                continue
                
            # Create mask for valid ranks (not NaN)
            valid_mask = ~np.isnan(ranks)
            if not np.any(valid_mask):
                continue
                
            # Store valid indices, ranks, and IDs
            valid_indices = np.where(valid_mask)[0]
            valid_ranks = ranks[valid_mask]
            valid_ids = np.array([ids[i] for i in valid_indices])
            
            # Store in cache
            measurement_cache[key] = {
                'valid_ranks': valid_ranks,
                'valid_ids': valid_ids,
                'sample_id': sample_id
            }
        
        # Main permutation loop
        permuted_results = []
        # Lists to store mean and median for each permutation
        perm_means = []
        perm_medians = []
        
        for _ in tqdm(range(n_permutations), desc="Processing permutations"):
            # Generate permutations for each sample
            sample_permutations = {}
            for sample_id, data in sample_lookup.items():
                indices = np.random.permutation(len(data['ids']))
                sample_permutations[sample_id] = dict(zip(data['ids'], data['values'][indices]))
            
            # Process measurements using cached data
            perm_result = {}
            for key, cache in measurement_cache.items():
                sample_id = cache['sample_id']
                valid_ranks = cache['valid_ranks']
                valid_ids = cache['valid_ids']
                
                # Get permuted values for valid IDs
                permuted_values = np.array([sample_permutations[sample_id].get(id_val, None) 
                                        for id_val in valid_ids])
                
                # Remove any None values that couldn't be mapped
                good_indices = permuted_values != None
                if not np.any(good_indices):
                    continue
                    
                # Calculate contrast with valid data only
                perm_result[key] = contrast_effect(
                    valid_ranks[good_indices],
                    permuted_values[good_indices],
                    self.control_group_name
                )
            
            # Calculate mean and median for this permutation if we have values
            if perm_result:
                contrast_values = list(perm_result.values())
                perm_means.append(np.mean(contrast_values))
                perm_medians.append(np.median(contrast_values))
            
            permuted_results.append(perm_result)
        
        self.permuted_results = permuted_results
        self.perm_means = perm_means
        self.perm_medians = perm_medians
        
        # Calculate original mean and median contrasts
        original_contrasts = [v.get("between_group_contrast") for v in valid_measurements.values() 
                            if v.get("between_group_contrast") is not None]
        
        if original_contrasts:
            self.original_mean_contrast = np.mean(original_contrasts)
            self.original_median_contrast = np.median(original_contrasts)
        else:
            self.original_mean_contrast = None
            self.original_median_contrast = None
        
        return permuted_results

    def regression(self, formula, target_nodes=None, add_network_categories=False,
                filter=None, calculate_permutation_stats=True, n_permutations=1000, 
                seed=None, confidence_level=0.95, column="group"):
        """
        Performs regression analysis on specification data with optional permutation-based inference.
        
        Parameters:
            formula (str): R-style formula for regression (e.g., "~ band + band:eyes")
            target_nodes (tuple, optional): Specific node pair to analyze. If None, analyzes all node pairs.
            add_network_categories (bool): Whether to add network relationship categories.
            filter (tuple, optional): A tuple of ("include", dict) or ("exclude", dict) to filter data.
            calculate_permutation_stats (bool): Whether to derive p-values from permutation tests (default: True).
            n_permutations (int): Number of permutations to run if permutation stats are requested.
                If existing permutation data has fewer permutations, additional permutations will be generated.
            seed (int, optional): Random seed for permutation.
            confidence_level (float): Confidence level for intervals (default: 0.95).
            column (str): Column to use for permutation tests, default is "group"
        
        Returns:
            pd.DataFrame, pd.DataFrame, pd.DataFrame: Three DataFrames containing model summary, parameter estimates, and contrast statistics
        """
        # Get specification data
        spec_df, results_df = self.specification_data(
            target_nodes=target_nodes,
            add_network_categories=add_network_categories,
            filter=filter
        )
        
        if len(spec_df) == 0 or len(results_df) == 0:
            print("No data available for regression analysis.")
            return None, None, None
        
        # Merge the dataframes
        combined_df = pd.concat([spec_df, results_df], axis=1)
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        
        # Ensure the formula starts with the dependent variable
        if not formula.startswith("contrast"):
            if formula.startswith("~"):
                formula = "contrast " + formula
            else:
                formula = "contrast ~ " + formula
        
        # Fit the model
        try:
            model = smf.ols(formula=formula, data=combined_df)
            result = model.fit()
        except Exception as e:
            print(f"Error fitting regression model: {str(e)}")
            return None, None, None
        
        # Create summary DataFrame
        model_summary = pd.DataFrame({
            'Statistic': ['Formula', 'Observations', 'R-squared', 'Adj. R-squared', 'F-statistic'],
            'Value': [formula, result.nobs, result.rsquared, result.rsquared_adj, result.fvalue]
        })
        
        # Create parameter estimates DataFrame
        param_estimates = pd.DataFrame({
            'Parameter': result.params.index,
            'Estimate': result.params.values,
            'Std. Error': result.bse.values,
            't-value': result.tvalues.values
        })
        
        # Initialize contrast stats DataFrame
        contrast_stats = pd.DataFrame({
            'Statistic': ['Mean Contrast', 'Median Contrast'],
            'Value': [np.mean(results_df['contrast']), np.median(results_df['contrast'])]
        })
        
        # If permutation-based inference is requested
        if calculate_permutation_stats:
            # Check if we need to run or update permutations
            need_new_permutations = False
            
            # Case 1: No permutation results exist
            if not hasattr(self, "permuted_results") or self.permuted_results is None:
                need_new_permutations = True
            # Case 2: Existing permutations are fewer than requested
            elif len(self.permuted_results) < n_permutations:
                print(f"Found {len(self.permuted_results)} existing permutations, but {n_permutations} requested.")
                print(f"Running additional {n_permutations - len(self.permuted_results)} permutations...")
                need_new_permutations = True
            
            # Run new permutations if needed
            if need_new_permutations:
                if hasattr(self, "permuted_results") and self.permuted_results and len(self.permuted_results) < n_permutations:
                    # Calculate how many more permutations we need
                    additional_permutations = n_permutations - len(self.permuted_results)
                    
                    # Run additional permutations and combine with existing ones
                    print(f"Running {additional_permutations} additional permutations...")
                    new_permutations = self.permute(n_permutations=additional_permutations, seed=seed, column=column)
                    
                    # Combine existing and new permutations
                    self.permuted_results.extend(new_permutations)
                    
                    # Ensure we only keep the requested number of permutations
                    if len(self.permuted_results) > n_permutations:
                        self.permuted_results = self.permuted_results[:n_permutations]
                else:
                    # No permutations or need to start from scratch
                    print(f"Running permutation test with {n_permutations} permutations...")
                    self.permute(n_permutations=n_permutations, seed=seed, column=column)
            
            # Calculate permutation-based statistics if we have permutation results
            if hasattr(self, "permuted_results") and self.permuted_results:
                print("Calculating permutation-based statistics...")
                
                # Limit the number of permutations to the requested number
                used_permutations = self.permuted_results[:n_permutations]
                
                # Collect permutation model results
                perm_f_values = []
                perm_t_values = {param: [] for param in result.params.index}
                perm_params = {param: [] for param in result.params.index}  # Store parameter estimates too
                
                # Convert permutation results to DataFrame format for regression
                for perm_result in used_permutations:
                    # Extract permuted contrasts in the same order
                    perm_contrasts = []
                    for name in results_df["name"]:
                        if name in perm_result:
                            perm_contrasts.append(perm_result[name])
                        else:
                            # Skip this permutation if data is missing
                            break
                    
                    # Skip if some data was missing
                    if len(perm_contrasts) != len(results_df):
                        continue
                    
                    # Create permuted dataframe
                    perm_results_df = results_df.copy()
                    perm_results_df["contrast"] = perm_contrasts
                    
                    # Merge with specification data
                    perm_combined_df = pd.concat([spec_df, perm_results_df], axis=1)
                    perm_combined_df = perm_combined_df.loc[:, ~perm_combined_df.columns.duplicated()]
                    
                    # Fit the model on permuted data
                    try:
                        perm_model = smf.ols(formula=formula, data=perm_combined_df)
                        perm_result_obj = perm_model.fit()
                        
                        # Store F-statistic
                        perm_f_values.append(perm_result_obj.fvalue)
                        
                        # Store parameter estimates and t-values for each parameter
                        for param in perm_t_values:
                            if param in perm_result_obj.tvalues:
                                perm_t_values[param].append(perm_result_obj.tvalues[param])
                                perm_params[param].append(perm_result_obj.params[param])
                    except Exception:
                        # Skip this permutation if regression fails
                        continue
                
                # Store the permutation data for use in print_apa_format
                self._last_regression_perm_data = {
                    "perm_f_values": perm_f_values,
                    "perm_t_values": perm_t_values,
                    "perm_params": perm_params
                }
                
                # Calculate permutation-based p-values
                if perm_f_values:
                    # F-statistic p-value
                    perm_f_pvalue = sum(f >= result.fvalue for f in perm_f_values) / len(perm_f_values)
                    
                    # Add F p-value to model summary
                    f_pvalue_df = pd.DataFrame({
                        'Statistic': ['F p-value (perm)', 'Permutations used'], 
                        'Value': [perm_f_pvalue, len(perm_f_values)]
                    })
                    model_summary = pd.concat([model_summary, f_pvalue_df], ignore_index=True)
                    
                    # Parameter t-values p-values
                    p_values = []
                    
                    for param in result.params.index:
                        if param in perm_t_values and perm_t_values[param]:
                            # Two-tailed test: count permutation t-values more extreme than observed
                            t_value = result.tvalues[param]
                            perm_t_pvalue = sum(abs(t) >= abs(t_value) for t in perm_t_values[param]) / len(perm_t_values[param])
                            p_values.append(perm_t_pvalue)
                        else:
                            p_values.append(np.nan)
                    
                    # Add p-values to parameter table
                    param_estimates['p-value (perm)'] = p_values
                    
                    # Add note about which column was permuted
                    note_df = pd.DataFrame({
                        'Statistic': ['Permuted column'], 
                        'Value': [column]
                    })
                    model_summary = pd.concat([model_summary, note_df], ignore_index=True)
                    
                else:
                    print("Warning: No valid permutation results for statistical inference.")
                    
                    # Add note
                    note_df = pd.DataFrame({
                        'Statistic': ['Note'], 
                        'Value': ["Permutation test yielded insufficient valid results"]
                    })
                    model_summary = pd.concat([model_summary, note_df], ignore_index=True)
                
                # Add mean and median contrast statistics with confidence intervals
                if hasattr(self, "perm_means") and self.perm_means and hasattr(self, "perm_medians") and self.perm_medians:
                    # Calculate 95% confidence intervals for mean and median
                    alpha = 1 - confidence_level
                    perm_means_sorted = sorted(self.perm_means)
                    perm_medians_sorted = sorted(self.perm_medians)
                    
                    lower_idx = int(alpha/2 * len(perm_means_sorted))
                    upper_idx = int((1 - alpha/2) * len(perm_means_sorted))
                    
                    mean_ci_lower = perm_means_sorted[max(0, lower_idx)]
                    mean_ci_upper = perm_means_sorted[min(len(perm_means_sorted)-1, upper_idx)]
                    
                    median_ci_lower = perm_medians_sorted[max(0, lower_idx)]
                    median_ci_upper = perm_medians_sorted[min(len(perm_medians_sorted)-1, upper_idx)]
                    
                    # Calculate p-values for mean and median
                    if hasattr(self, "original_mean_contrast") and self.original_mean_contrast is not None:
                        mean_pvalue = sum(abs(m) >= abs(self.original_mean_contrast) for m in self.perm_means) / len(self.perm_means)
                    else:
                        mean_pvalue = None
                    
                    if hasattr(self, "original_median_contrast") and self.original_median_contrast is not None:
                        median_pvalue = sum(abs(m) >= abs(self.original_median_contrast) for m in self.perm_medians) / len(self.perm_medians)
                    else:
                        median_pvalue = None
                    
                    # Add to contrast_stats DataFrame
                    ci_df = pd.DataFrame({
                        'Statistic': [
                            'Mean CI Lower', 'Mean CI Upper', 'Mean p-value',
                            'Median CI Lower', 'Median CI Upper', 'Median p-value'
                        ], 
                        'Value': [
                            mean_ci_lower, mean_ci_upper, mean_pvalue,
                            median_ci_lower, median_ci_upper, median_pvalue
                        ]
                    })
                    contrast_stats = pd.concat([contrast_stats, ci_df], ignore_index=True)
                    
            else:
                print("Warning: Permutation test failed or was not run.")
                
                # Add note
                note_df = pd.DataFrame({
                    'Statistic': ['Note'], 
                    'Value': ["Permutation test failed"]
                })
                model_summary = pd.concat([model_summary, note_df], ignore_index=True)
        
        return model_summary, param_estimates, contrast_stats

    def print_apa_format(self, regression_results, alpha=0.05, digits=3):
        """
        Formats regression results in APA style, with standard errors and confidence 
        intervals derived from permutation distributions when available.
        
        Parameters:
            regression_results (tuple): The (model_summary, param_estimates, contrast_stats) tuple returned by the regression method
            alpha (float): Significance level for highlighting significant results (default: 0.05)
            digits (int): Number of decimal places to display (default: 3)
            
        Returns:
            str: Formatted APA style text output
        """
        import numpy as np
        import re
        from helpers_functions import clean_parameter_name, format_significance_stars
        from helpers_functions import calculate_confidence_interval, get_model_info, format_parameter_row, get_permutation_se
        
        # Unpack the regression results
        if len(regression_results) == 3:
            model_summary, param_estimates, contrast_stats = regression_results
        else:
            model_summary, param_estimates = regression_results
            contrast_stats = None
        
        if model_summary is None or param_estimates is None:
            return "No valid regression results to format."
        
        # Get model information using helper function
        formula = get_model_info(model_summary, 'Formula', 'Unknown')
        r_squared = get_model_info(model_summary, 'R-squared', 0.0)
        f_stat = get_model_info(model_summary, 'F-statistic', 0.0)
        
        # Check if permutation p-values are available
        has_perm_pvalues = 'F p-value (perm)' in model_summary['Statistic'].values
        
        if has_perm_pvalues:
            f_pvalue = get_model_info(model_summary, 'F p-value (perm)', None)
            p_value_type = "permutation-based"
            
            # Check which column was permuted
            if 'Permuted column' in model_summary['Statistic'].values:
                permuted_column = get_model_info(model_summary, 'Permuted column', 'unknown')
                p_value_type = f"permutation-based (permuted {permuted_column})"
        else:
            f_pvalue = None
            p_value_type = None
        
        # Format the values
        r_squared_fmt = f"{r_squared:.{digits}f}"
        f_stat_fmt = f"{f_stat:.{digits}f}"
        f_pvalue_fmt = f"{f_pvalue:.{digits}f}" if f_pvalue is not None else "N/A"
        
        # Start building the output string
        apa_output = []
        
        # Add model summary section
        apa_output.append(f"Regression Analysis Results (APA Format)")
        apa_output.append(f"----------------------------------------")
        apa_output.append(f"Model: {formula}")
        apa_output.append(f"")
        
        # Add overall model statistics
        apa_output.append(f"The regression model explained {r_squared_fmt} of the variance, "
                        f"F = {f_stat_fmt}, "
                        f"{'' if f_pvalue is None else 'p = ' + f_pvalue_fmt}")
        
        if p_value_type:
            apa_output.append(f"Note: p-values are {p_value_type}")
        
        # Add mean and median contrast information if available
        if contrast_stats is not None:
            apa_output.append(f"")
            apa_output.append(f"Contrast Statistics:")
            
            # Extract mean contrast and CI
            mean_contrast = get_model_info(contrast_stats, 'Mean Contrast', None)
            mean_ci_lower = get_model_info(contrast_stats, 'Mean CI Lower', None)
            mean_ci_upper = get_model_info(contrast_stats, 'Mean CI Upper', None)
            mean_pvalue = get_model_info(contrast_stats, 'Mean p-value', None)
            
            # Extract median contrast and CI
            median_contrast = get_model_info(contrast_stats, 'Median Contrast', None)
            median_ci_lower = get_model_info(contrast_stats, 'Median CI Lower', None)
            median_ci_upper = get_model_info(contrast_stats, 'Median CI Upper', None)
            median_pvalue = get_model_info(contrast_stats, 'Median p-value', None)
            
            # Format mean contrast
            if mean_contrast is not None:
                mean_stars = format_significance_stars(mean_pvalue) if mean_pvalue is not None else ""
                mean_fmt = f"{mean_contrast:.{digits}f}{mean_stars}"
                
                if mean_ci_lower is not None and mean_ci_upper is not None:
                    apa_output.append(f"Mean Contrast: {mean_fmt}, 95% CI [{mean_ci_lower:.{digits}f}, {mean_ci_upper:.{digits}f}]" + 
                                    (f", p = {mean_pvalue:.{digits}f}" if mean_pvalue is not None else ""))
                else:
                    apa_output.append(f"Mean Contrast: {mean_fmt}")
            
            # Format median contrast
            if median_contrast is not None:
                median_stars = format_significance_stars(median_pvalue) if median_pvalue is not None else ""
                median_fmt = f"{median_contrast:.{digits}f}{median_stars}"
                
                if median_ci_lower is not None and median_ci_upper is not None:
                    apa_output.append(f"Median Contrast: {median_fmt}, 95% CI [{median_ci_lower:.{digits}f}, {median_ci_upper:.{digits}f}]" + 
                                    (f", p = {median_pvalue:.{digits}f}" if median_pvalue is not None else ""))
                else:
                    apa_output.append(f"Median Contrast: {median_fmt}")
        
        apa_output.append(f"")
        
        # Add regression coefficients table header
        apa_output.append(f"Regression Coefficients:")
        apa_output.append(f"{'Parameter':<20} {'Estimate':<10} {'SE':<10} {'t':<10}" + 
                        (f" {'p':<10}" if 'p-value (perm)' in param_estimates.columns else "") +
                        (f" {'95% CI':<20}" if '95% CI Lower' in param_estimates.columns else ""))
        apa_output.append(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10}" + 
                        (f" {'-'*10}" if 'p-value (perm)' in param_estimates.columns else "") +
                        (f" {'-'*20}" if '95% CI Lower' in param_estimates.columns else ""))
        
        # Calculate permutation-based standard errors if we have the raw permutation data
        perm_se = {}
        if hasattr(self, "permuted_results") and self.permuted_results and has_perm_pvalues:
            # Access the permutation data collected during regression analysis
            if hasattr(self, "_last_regression_perm_data") and self._last_regression_perm_data:
                perm_t_values = self._last_regression_perm_data.get("perm_t_values", {})
                perm_params = self._last_regression_perm_data.get("perm_params", {})
                
                for param in param_estimates['Parameter']:
                    se = get_permutation_se(param, param_estimates, perm_params, perm_t_values)
                    if se is not None:
                        perm_se[param] = se
        
        # Add individual parameters
        for _, row in param_estimates.iterrows():
            param_name = row['Parameter']
            estimate = row['Estimate']
            
            # Use permutation-based SE if available
            std_err = perm_se.get(param_name, None)
            t_value = row['t-value']
            
            # Get p-value if available
            p_value = row.get('p-value (perm)', None)
            
            # Calculate confidence interval
            ci_fmt = None
            if param_name in perm_se:
                ci_fmt = calculate_confidence_interval(estimate, std_err, digits=digits)
            elif '95% CI Lower' in row and '95% CI Upper' in row:
                ci_lower = row['95% CI Lower']
                ci_upper = row['95% CI Upper']
                ci_fmt = f"[{ci_lower:.{digits}f}, {ci_upper:.{digits}f}]"
            
            # Format the parameter row using the helper function
            param_line = format_parameter_row(
                param_name=param_name,
                estimate=estimate,
                std_err=std_err,
                t_value=t_value,
                p_value=p_value,
                ci_fmt=ci_fmt,
                digits=digits
            )
            
            apa_output.append(param_line)
        
        # Add significance code explanation if p-values are present
        if 'p-value (perm)' in param_estimates.columns:
            apa_output.append(f"")
            apa_output.append(f"Significance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        
        # Add note about permutation basis if applicable
        if p_value_type:
            if perm_se:
                apa_output.append(f"")
                apa_output.append(f"Note: All p-values, standard errors, and confidence intervals are {p_value_type}.")
            else:
                apa_output.append(f"")
                apa_output.append(f"Note: p-values are {p_value_type}.")
        
        # Join and return the formatted output
        return "\n".join(apa_output)

    def summary(self):
        """
        Print a summary of the measurement conditions in the study.
        Shows all condition keys and their possible values across all measurements.
        Nodes are grouped by their labels and displayed in a hierarchical format.
        If permutations have been run, includes mean and median contrast statistics.
        """
        if not self.data:
            print("No data loaded in this study.")
            return
        
        print(f"Study summary: {len(self.data)} measurements loaded")
        
        # Group and display nodes by labels
        if self.node_labels:
            # Create a dictionary to group nodes by label
            nodes_by_label = {}
            for node, label in self.node_labels.items():
                if node in self.node_list:  # Only include nodes that are in node_list
                    if label not in nodes_by_label:
                        nodes_by_label[label] = []
                    nodes_by_label[label].append(node)
            
            # Print nodes grouped by labels
            print("\nNodes by label:")
            for label, nodes in sorted(nodes_by_label.items()):
                print(f"{label}:")
                print(f"       {', '.join(sorted(nodes))}")
        else:
            # If no labels, just print the sorted node list
            print(f"\nNodes: {', '.join(sorted(self.node_list))}")
        
        # Collect all condition keys and their possible values
        conditions_summary = {}
        
        for name, measurement in self.data.items():
            if "measurement_conditions" in measurement and measurement["measurement_conditions"]:
                for key, value in measurement["measurement_conditions"].items():
                    if key not in conditions_summary:
                        conditions_summary[key] = set()
                    conditions_summary[key].add(value)
        
        # Print condition keys and their possible values
        if conditions_summary:
            print("\nMeasurement conditions:")
            for key, values in sorted(conditions_summary.items()):
                print(f"  {key}: {', '.join(sorted(values))}")
        else:
            print("\nNo measurement conditions found.")
        
        # Print sample information
        if self.samples:
            print("\nSamples:")
            for sample_id, sample_df in self.samples.items():
                # Count unique subjects in each sample
                unique_subjects = sample_df['ID'].nunique()
                print(f"  Sample ID: {sample_id}, Subjects: {unique_subjects}")
        else:
            print("\nNo sample information available.")
        
        # Print permutation statistics if available
        if hasattr(self, "permuted_results") and self.permuted_results:
            print("\nPermutation Statistics:")
            
            # Calculate and display original mean and median contrasts
            original_contrasts = []
            for name, measurement in self.data.items():
                if "between_group_contrast" in measurement and measurement["between_group_contrast"] is not None:
                    original_contrasts.append(measurement["between_group_contrast"])
            
            if original_contrasts:
                mean_contrast = np.mean(original_contrasts)
                median_contrast = np.median(original_contrasts)
                
                print(f"  Original Mean Contrast: {mean_contrast:.4f}")
                print(f"  Original Median Contrast: {median_contrast:.4f}")
            
            # Display permutation-based statistics
            if hasattr(self, "perm_means") and self.perm_means and hasattr(self, "perm_medians") and self.perm_medians:
                # Calculate 95% confidence intervals
                perm_means_sorted = sorted(self.perm_means)
                perm_medians_sorted = sorted(self.perm_medians)
                
                lower_idx = int(0.025 * len(perm_means_sorted))
                upper_idx = int(0.975 * len(perm_means_sorted))
                
                mean_ci_lower = perm_means_sorted[max(0, lower_idx)]
                mean_ci_upper = perm_means_sorted[min(len(perm_means_sorted)-1, upper_idx)]
                
                median_ci_lower = perm_medians_sorted[max(0, lower_idx)]
                median_ci_upper = perm_medians_sorted[min(len(perm_medians_sorted)-1, upper_idx)]
                
                # Calculate p-values
                if hasattr(self, "original_mean_contrast") and self.original_mean_contrast is not None:
                    mean_pvalue = sum(abs(m) >= abs(self.original_mean_contrast) for m in self.perm_means) / len(self.perm_means)
                    print(f"  Mean Contrast p-value: {mean_pvalue:.4f}")
                
                if hasattr(self, "original_median_contrast") and self.original_median_contrast is not None:
                    median_pvalue = sum(abs(m) >= abs(self.original_median_contrast) for m in self.perm_medians) / len(self.perm_medians)
                    print(f"  Median Contrast p-value: {median_pvalue:.4f}")
                
                print(f"  Mean Contrast 95% CI: [{mean_ci_lower:.4f}, {mean_ci_upper:.4f}]")
                print(f"  Median Contrast 95% CI: [{median_ci_lower:.4f}, {median_ci_upper:.4f}]")
                print(f"  Number of permutations: {len(self.perm_means)}")

    def merge_independent_condition(self, ignore_conditions=None):
        """
        Creates a new Study object with measurements merged across the specified conditions to ignore.
        Merges corresponding measurements across different samples while preserving unique subjects.
        
        Parameters:
            ignore_conditions (list): List of condition keys that should be ignored for merging
                    
        Returns:
            Study: A new Study object containing merged measurements
        """

        
        if ignore_conditions is None:
            ignore_conditions = []
        elif isinstance(ignore_conditions, str):
            ignore_conditions = [ignore_conditions]
        
        print(f"Starting merge with ignore_conditions: {ignore_conditions}")
        
        # Create a new Study object with same properties
        new_study = Study()
        new_study.node_list = self.node_list.copy()
        new_study.node_labels = deepcopy(self.node_labels) if self.node_labels else None
        new_study.control_group_name = self.control_group_name
        
        # Early exit if no data
        if not self.data:
            print("No data to merge.")
            return new_study
        
        # Pre-filter measurements
        valid_measurements = {}
        for name, item in self.data.items():
            if "measurement_conditions" in item and item["measurement_conditions"] and "nodes" in item:
                valid_measurements[name] = item
        
        if len(valid_measurements) < 2:
            print("Insufficient valid measurements for merging.")
            return new_study
        
        # Group measurements by node and non-ignored conditions
        measurement_groups = {}
        
        for name, item in valid_measurements.items():
            # Create a key from nodes and non-ignored conditions
            nodes = tuple(item["nodes"])
            
            # Extract relevant conditions (excluding ignored ones)
            other_conditions = {k: v for k, v in item["measurement_conditions"].items() 
                            if k not in ignore_conditions}
            
            # Convert dict to frozenset of tuples for hashing
            other_conditions_key = frozenset(other_conditions.items())
            
            # Create a group key
            group_key = (nodes, other_conditions_key)
            
            # Add to grouping dictionary
            if group_key not in measurement_groups:
                measurement_groups[group_key] = []
            
            measurement_groups[group_key].append(name)
        
        # Keep only groups with multiple measurements
        multi_measurement_groups = {k: v for k, v in measurement_groups.items() if len(v) > 1}
        print(f"Found {len(multi_measurement_groups)} groups with multiple measurements")
        
        # For each group, create a merged measurement
        merged_counter = 0
        
        # Create one merged sample for all merges
        merged_sample_id = "merged_sample"
        
        # Track unique subjects with their information
        unique_subjects = {}  # {merged_id: {'group': group, 'original_sample': sample_id, 'original_id': id_str}}
        
        # Process each measurement group
        for group_key, group_names in tqdm(multi_measurement_groups.items(), desc="Merging measurements"):
            try:
                # Get example measurement for attribute extraction
                example_name = group_names[0]
                example_data = valid_measurements[example_name]
                nodes = example_data["nodes"]
                
                # Create merged name
                merged_name = f"merged_{merged_counter}"
                merged_counter += 1
                
                # Keep non-ignored conditions
                other_conditions = {k: v for k, v in example_data.get("measurement_conditions", {}).items() 
                                if k not in ignore_conditions}
                
                # Create merged condition values
                merged_condition_values = {}
                for cond in ignore_conditions:
                    values = set()
                    for name in group_names:
                        if cond in valid_measurements[name].get("measurement_conditions", {}):
                            values.add(valid_measurements[name]["measurement_conditions"][cond])
                    
                    if values:
                        merged_condition_values[cond] = f"merged({','.join(sorted(values))})"
                
                # Initialize merged data
                merged_item = {
                    "measurement_conditions": other_conditions.copy(),
                    "nodes": nodes,
                    "id": [],
                    "sample_id": merged_sample_id,
                    "values": [],
                    "ranks": None,
                    "merged_from": group_names,
                    "ignored_conditions": ignore_conditions
                }
                
                # Add merged values for ignored conditions
                merged_item["measurement_conditions"].update(merged_condition_values)
                
                # Group measurements by sample
                measurements_by_sample = {}
                for name in group_names:
                    sample_id = valid_measurements[name].get("sample_id", "unknown")
                    if sample_id not in measurements_by_sample:
                        measurements_by_sample[sample_id] = []
                    measurements_by_sample[sample_id].append(name)
                
                # Process each sample and collect values for this measurement
                all_ids = []
                all_values = []
                all_groups = []
                
                for sample_id, sample_measurements in measurements_by_sample.items():
                    for meas_name in sample_measurements:
                        meas_data = valid_measurements[meas_name]
                        
                        if "id" in meas_data and "values" in meas_data and meas_data["id"] and meas_data["values"]:
                            # Get sample for group info
                            if sample_id in self.samples:
                                id_to_group = dict(zip(
                                    self.samples[sample_id]['ID'].astype(str), 
                                    self.samples[sample_id]['group']
                                ))
                            else:
                                id_to_group = {}
                            
                            # Process each subject
                            for i, id_val in enumerate(meas_data["id"]):
                                id_str = str(id_val)
                                
                                # Create unique ID for the merged sample
                                merged_id = f"{sample_id}_{id_str}"
                                
                                # Get value if available
                                if i < len(meas_data["values"]):
                                    value = meas_data["values"][i]
                                else:
                                    value = None
                                
                                # Get group
                                group = id_to_group.get(id_str, "unknown")
                                
                                # Add to measurement values
                                all_ids.append(merged_id)
                                all_values.append(value)
                                all_groups.append(group)
                                
                                # Track unique subjects (only once)
                                if merged_id not in unique_subjects:
                                    unique_subjects[merged_id] = {
                                        'group': group,
                                        'original_sample': sample_id,
                                        'original_id': id_str
                                    }
                
                # Store in merged item
                merged_item["id"] = all_ids
                merged_item["values"] = all_values
                
                # Calculate ranks and contrast
                if all_values and any(v is not None for v in all_values):
                    # Convert to array for calculations
                    values_array = np.array(all_values, dtype=float)
                    group_array = np.array(all_groups)
                    
                    # Calculate ranks and contrast
                    ranks = calculate_ranks(values_array)
                    merged_item["ranks"] = ranks
                    
                    merged_item["between_group_contrast"] = contrast_effect(
                        ranks, group_array, self.control_group_name
                    )
                
                # Add the merged item to the new study's data
                new_study.data[merged_name] = merged_item
                
            except Exception as e:
                print(f"Error processing group {group_names}: {str(e)}")
        
        # Create the merged sample DataFrame with unique subjects only
        if unique_subjects:
            merged_sample_data = {
                'ID': list(unique_subjects.keys()),
                'group': [s['group'] for s in unique_subjects.values()],
                'original_sample': [s['original_sample'] for s in unique_subjects.values()],
                'original_id': [s['original_id'] for s in unique_subjects.values()]
            }
            new_study.samples[merged_sample_id] = pd.DataFrame(merged_sample_data)
            
            print(f"Added {len(unique_subjects)} unique subjects to merged sample")
        
        print(f"Merging complete: {len(new_study.data)} measurements in new study")
        
        return new_study

    def save(self, file_path):
        """
        Save the current Study object to a file using pickle serialization.
        
        Parameters:
            file_path (str): Path where the Study object will be saved
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            with open(file_path, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
            print(f"Study saved successfully to {file_path}")
        except Exception as e:
            print(f"Error saving Study: {str(e)}")

    @classmethod
    def load(cls, file_path):
        """
        Load a Study object from a file.
        
        Parameters:
            file_path (str): Path to the saved Study object
            compressed (bool): Whether the file was saved with compression (default: False)
            
        Returns:
            Study: The loaded Study object or None if loading failed
        """
        try:
            with open(file_path, 'rb') as f:
                study = pickle.load(f)
            print(f"Study loaded successfully from {file_path}")
            return study
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except Exception as e:
            print(f"Error loading Study: {str(e)}")
            return None