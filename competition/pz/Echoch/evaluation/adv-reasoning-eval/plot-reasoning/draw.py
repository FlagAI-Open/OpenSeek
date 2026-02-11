import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker # Import ticker for formatting
import numpy as np # Import numpy for calculations
from scipy import stats # Import scipy.stats for linear regression

def plot_scatter_subplots(csv_file_path, output_filename="scatter_plots_1x4.png"):
    """
    Reads data from a CSV file and plots 1x4 scatter subplots
    with y-axis tick labels formatted to two decimal places,
    fixed y-axis limits and tick intervals for specific subplots,
    a light-colored dashed least squares regression line in each subplot (no grid/legend),
    and saves the plot to a file. The third point in the last subplot is excluded from its regression
    and highlighted in red. Subplots are arranged in a single row and made narrower.

    Args:
    csv_file_path (str): The path to the CSV file.
    output_filename (str): The name of the file to save the plot to.
    """
    try:
        # Matplotlib will use a default font.
        plt.rcParams['axes.unicode_minus'] = False  # Fix for displaying minus sign

        # Read the CSV file
        df = pd.read_csv(csv_file_path)

        # Get column names
        column_names = df.columns
        x_column_name = column_names[0]  # First column as x-axis

        # Check if there are enough columns for plotting
        if len(column_names) < 5:
            print(f"Error: The CSV file needs at least 5 columns of data, but found only {len(column_names)}.")
            print("The first column is used for the x-axis, and the next four columns are for the y-axes of the subplots.")
            return
        
        # Define y_column_names before using it in the check below
        y_column_names = column_names[1:5] # Columns 2 to 5 as y-axes
        
        # Check if enough data points for outlier removal in the last subplot
        # The last subplot corresponds to the 4th y-column, which is index 3 in a 0-indexed enumeration
        if len(df) < 3 and any(idx == 3 for idx, _ in enumerate(range(len(y_column_names)))): 
            print(f"Warning: Not enough data points to exclude the third point for regression in the last subplot.")


        # Create 1x4 subplots
        # Adjust figsize: width increased for 4 plots, height can be reduced to make them narrower,
        # or keep height and let aspect ratio change. Let's try (20, 5) for a wider, shorter figure.
        # You might need to adjust this based on your paper's column width.
        fig, axes = plt.subplots(1, 4, figsize=(12, 4.5)) # 1 row, 4 columns.
        # axes is already a 1D array if nrows or ncols is 1.

        # X-axis title
        x_axis_title = "Log Pre-Training Compute"
        # Y-axis title
        y_axis_title = "Scores"

        # Iterate through the four datasets and plot scatter graphs
        for i, y_col in enumerate(y_column_names):
            ax = axes[i] # Direct indexing for 1D array
            x_data_original = df[x_column_name]
            y_data_original = df[y_col]
            
            # Scatter plot all original data points
            ax.scatter(x_data_original, y_data_original, alpha=0.7, edgecolors='w', linewidth=0.5, s=30) # s=30 for slightly smaller points
            
            x_data_for_regression = x_data_original
            y_data_for_regression = y_data_original

            # For the last subplot (index 3), exclude the third point (index 2) from regression
            # and highlight it in red.
            if i == 3 and len(x_data_original) >= 3:
                outlier_x_val = x_data_original.iloc[2]
                outlier_y_val = y_data_original.iloc[2]
                
                # Data for regression excludes the outlier
                x_data_for_regression = x_data_original.drop(x_data_original.index[2]).reset_index(drop=True)
                y_data_for_regression = y_data_original.drop(y_data_original.index[2]).reset_index(drop=True)
                
                # Highlight the outlier point in red
                ax.scatter(outlier_x_val, outlier_y_val, color='red', s=40, zorder=5) # s=40, slightly adjusted


            # Perform linear regression on potentially filtered data
            if len(x_data_for_regression) > 1 and len(y_data_for_regression) > 1: # Need at least 2 points for regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_data_for_regression, y_data_for_regression)
                
                # Create points for the regression line based on the range of original x_data
                line_x_ends = np.array([x_data_original.min(), x_data_original.max()])
                line_y = slope * line_x_ends + intercept
                
                # Plot the regression line with a light color and dashed style, no label
                ax.plot(line_x_ends, line_y, color='lightcoral', linestyle='--', linewidth=1.5) # linewidth reduced
            else:
                print(f"Warning: Not enough data points for regression in subplot for '{y_col}' after filtering.")

            
            ax.set_xlabel(x_axis_title, fontsize=10) # Reduced font size
            ax.set_ylabel(y_axis_title, fontsize=10) # Reduced font size
            ax.set_title(f"{y_col}", fontsize=11)    # Reduced font size
            ax.tick_params(axis='both', which='major', labelsize=8) # Reduced tick label size
            
            # Format y-axis tick labels to two decimal places for all subplots
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            
            # Apply specific y-axis limits and tick intervals
            if i < 3:  # First three subplots
                ax.set_ylim([0.45, 0.75])
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            else:  # Last subplot (index 3)
                ax.set_ylim([0.20, 0.35])
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            

        # Adjust spacing between subplots
        # For 1x4 layout, tight_layout is crucial. You might also use fig.subplots_adjust
        plt.tight_layout(pad=1.0) # pad adjusted for potentially tighter layout
        # fig.suptitle("Data Scatter Plots (1x4 Layout with Light Regression Lines)", fontsize=16, y=1.03) # This line is removed

        # Save the figure
        try:
            plt.savefig(output_filename, dpi=300, bbox_inches='tight') 
            print(f"Plot saved as '{output_filename}'")
        except Exception as e:
            print(f"Error saving plot: {e}")

        # Display the plot
        plt.show()

    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found. Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Please replace 'your_file.csv' with the path to your CSV file
    
    # --- Create dummy CSV file (start) ---
    import os
    dummy_csv_filename = "dummy_data_en.csv" 
    if not os.path.exists(dummy_csv_filename):
        np.random.seed(42) # for reproducibility
        num_points = 20
        log_compute = np.linspace(1, 10, num_points)
        
        data = {
            'log_pre_training_compute': log_compute,
            # Data for y-axis 0.45 to 0.75
            'Model_A_Perf': 0.02 * log_compute + 0.5 + np.random.normal(0, 0.03, num_points), 
            'Model_B_Score': -0.015 * log_compute + 0.65 + np.random.normal(0, 0.04, num_points), 
            'Algo_C_Effic': 0.01 * log_compute + 0.55 + np.random.normal(0, 0.05, num_points), 
            # Data for y-axis 0.20 to 0.35 for Algo_D_Acc
            'Algo_D_Acc': 0.012 * log_compute + 0.22 + np.random.normal(0, 0.015, num_points) 
        }
        
        # Introduce an outlier in the 3rd data point (index 2) of 'Algo_D_Acc'
        if num_points >= 3:
            data['Algo_D_Acc'][2] = 0.34 # Make it a clear high outlier for the specified y-axis range [0.20, 0.35]

        # Ensure data stays within specified plot ranges for better visualization of fixed axes
        data['Model_A_Perf'] = np.clip(data['Model_A_Perf'], 0.46, 0.74)
        data['Model_B_Score'] = np.clip(data['Model_B_Score'], 0.46, 0.74)
        data['Algo_C_Effic'] = np.clip(data['Algo_C_Effic'], 0.46, 0.74)
        
        temp_outlier_val = data['Algo_D_Acc'][2] if num_points >=3 else None 
        data['Algo_D_Acc'] = np.clip(data['Algo_D_Acc'], 0.21, 0.34) 
        if num_points >=3 and temp_outlier_val is not None: 
             data['Algo_D_Acc'][2] = temp_outlier_val if temp_outlier_val <= 0.35 else 0.34


        dummy_df = pd.DataFrame(data)
        dummy_df.to_csv(dummy_csv_filename, index=False)
        print(f"Created dummy data file '{dummy_csv_filename}' for demonstration.")
        csv_file_path_to_plot = dummy_csv_filename
    else:
        print(f"Dummy data file '{dummy_csv_filename}' already exists.")
        csv_file_path_to_plot = dummy_csv_filename
    # --- Create dummy CSV file (end) ---

    # Plot using the dummy data file and save it
    plot_scatter_subplots(csv_file_path_to_plot, output_filename="my_plots_1x4_narrow.png")
    
    # When you have your own CSV file, use the line below:
    # plot_scatter_subplots('your_file.csv', output_filename='your_plot_name.png')
