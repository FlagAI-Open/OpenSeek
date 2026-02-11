import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker # For formatting ticks
import numpy as np # For numerical calculations
import scipy.stats as stats # For linear regression
import os # For checking if dummy files exist

def plot_dual_scatter_subplots(csv_file_path1, csv_file_path2, output_filename="dual_scatter_plots_1x4.png"):
    """
    Reads data from two CSV files and plots 1x4 scatter subplots,
    each subplot containing two sets of data and their regression lines.

    Args:
    csv_file_path1 (str): Path to the first CSV file.
    csv_file_path2 (str): Path to the second CSV file.
    output_filename (str): Filename for the saved plot.
    """
    try:
        plt.rcParams['axes.unicode_minus'] = False  # Fix for displaying minus sign

        # Read the two CSV files
        df1 = pd.read_csv(csv_file_path1)
        df2 = pd.read_csv(csv_file_path2)

        # Get column names (assuming both files have the same structure and x-axis column name)
        column_names = df1.columns
        x_column_name = column_names[0]  # First column as x-axis

        # Check if there are enough columns for plotting
        if len(column_names) < 5:
            print(f"Error: CSV file '{csv_file_path1}' needs at least 5 columns, but found only {len(column_names)}.")
            return
        if len(df2.columns) < 5:
            print(f"Error: CSV file '{csv_file_path2}' needs at least 5 columns, but found only {len(df2.columns)}.")
            return

        y_column_names = column_names[1:5] # Columns 2 to 5 as y-axes

        # Create 1x4 subplots
        fig, axes = plt.subplots(1, 4, figsize=(12, 4.5)) # 1 row, 4 columns, adjusted size for narrower subplots

        # X-axis and Y-axis titles
        x_axis_title = "Log Pre-Training Compute"
        y_axis_title = "Scores"
        
        # Define colors and labels for the two datasets
        color1 = 'royalblue'
        label1 = 'w/ cot synthesis' # You can change this based on your data
        color2 = 'forestgreen'
        label2 = 'w/o cot synthesis' # You can change this based on your data

        # Iterate through the four y-column datasets and plot scatter graphs
        for i, y_col in enumerate(y_column_names):
            ax = axes[i]
            
            # --- Process Dataset 1 ---
            if y_col in df1.columns:
                x_data1_original = df1[x_column_name]
                y_data1_original = df1[y_col]
                
                # Scatter plot - Dataset 1
                # Add label only for the first subplot to avoid duplicate legend entries if using a figure-level legend
                # However, for per-subplot legends, label each time.
                ax.scatter(x_data1_original, y_data1_original, color=color1, alpha=0.7, edgecolors='w', linewidth=0.5, s=30, label=label1) 
                
                # Linear regression - Dataset 1
                if len(x_data1_original) > 1 and len(y_data1_original) > 1:
                    slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x_data1_original, y_data1_original)
                    line_x1_ends = np.array([x_data1_original.min(), x_data1_original.max()])
                    line_y1 = slope1 * line_x1_ends + intercept1
                    ax.plot(line_x1_ends, line_y1, color=color1, linestyle='--', linewidth=1.5)
            else:
                print(f"Warning: Column '{y_col}' not found in file '{csv_file_path1}'.")

            # --- Process Dataset 2 ---
            if y_col in df2.columns:
                x_data2_original = df2[x_column_name]
                y_data2_original = df2[y_col]

                # Scatter plot - Dataset 2
                ax.scatter(x_data2_original, y_data2_original, color=color2, alpha=0.7, edgecolors='w', linewidth=0.5, s=30, label=label2)

                # Linear regression - Dataset 2
                if len(x_data2_original) > 1 and len(y_data2_original) > 1:
                    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x_data2_original, y_data2_original)
                    line_x2_ends = np.array([x_data2_original.min(), x_data2_original.max()])
                    line_y2 = slope2 * line_x2_ends + intercept2
                    ax.plot(line_x2_ends, line_y2, color=color2, linestyle='--', linewidth=1.5)
            else:
                print(f"Warning: Column '{y_col}' not found in file '{csv_file_path2}'.")

            # Set axis labels, title, and tick parameters
            ax.set_xlabel(x_axis_title, fontsize=10)
            ax.set_ylabel(y_axis_title, fontsize=10)
            ax.set_title(f"{y_col}", fontsize=11)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Format y-axis tick labels to two decimal places
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            
            # Apply specific y-axis limits and tick intervals
            if i < 3:  # First three subplots
                ax.set_ylim([0.45, 0.70]) # Updated Y-limit
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            else:  # Last subplot
                ax.set_ylim([0.15, 0.40]) # Updated Y-limit
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            
            # Add legend to each subplot
            handles, labels_from_ax = ax.get_legend_handles_labels()
            if handles: # Only show legend if there are items to legend
                 ax.legend(fontsize=8, loc='lower right')


        # Adjust spacing between subplots
        plt.tight_layout(pad=1.0) 

        # Save the figure
        try:
            plt.savefig(output_filename, dpi=300, bbox_inches='tight') 
            print(f"Plot saved as '{output_filename}'")
        except Exception as e: # Catch specific exception for saving
            print(f"Error saving plot: {e}")

        # Display the plot
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: File not found. Please check file paths. {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def create_dummy_csv(filename, num_points=20, y_col_config=None, base_seed_offset=0):
    """Helper function to create dummy CSV files with specified ranges."""
    # Generate a seed based on filename and an offset to ensure different datasets
    seed = int(hash(filename[:10]) % (2**32 - 1)) + base_seed_offset 
    np.random.seed(seed)
    
    log_compute = np.linspace(1, 10, num_points)
    data = {'log_pre_training_compute': log_compute}

    # Updated default Y column configurations to match new Y-axis limits
    default_y_col_config = {
        'Y_Col1': {'base': 0.50, 'trend': 0.010, 'noise': 0.02, 'clip_min': 0.46, 'clip_max': 0.69},
        'Y_Col2': {'base': 0.55, 'trend': -0.005, 'noise': 0.03, 'clip_min': 0.46, 'clip_max': 0.69},
        'Y_Col3': {'base': 0.52, 'trend': 0.008, 'noise': 0.04, 'clip_min': 0.46, 'clip_max': 0.69},
        'Y_Col4': {'base': 0.20, 'trend': 0.010, 'noise': 0.02, 'clip_min': 0.16, 'clip_max': 0.39}
    }
    current_config = y_col_config if y_col_config is not None else default_y_col_config
        
    for i_col in range(1, 5):
        col_name_key = f'Y_Col{i_col}'
        # Use actual column names from config if provided, otherwise generate Y_Col1, Y_Col2 etc.
        # For simplicity, we assume dummy CSVs will have Y_Col1 to Y_Col4
        actual_col_name = col_name_key 
        
        config = current_config.get(col_name_key)
        if config:
            y_values = config['trend'] * log_compute + config['base'] + np.random.normal(0, config['noise'], num_points)
            data[actual_col_name] = np.clip(y_values, config['clip_min'], config['clip_max'])
        else: 
             data[actual_col_name] = np.random.rand(num_points) * (config['clip_max'] - config['clip_min']) + config['clip_min']


    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Created dummy data file '{filename}'.")


if __name__ == '__main__':
    # Create dummy data files (if they don't exist)
    dummy_csv_file1 = "data1.csv"
    dummy_csv_file2 = "data2.csv"

    # Define configurations for dummy data to fit new Y-axis ranges
    y_col_config1 = {
        'Y_Col1': {'base': 0.50, 'trend': 0.010, 'noise': 0.02, 'clip_min': 0.46, 'clip_max': 0.69},
        'Y_Col2': {'base': 0.55, 'trend': -0.005, 'noise': 0.03, 'clip_min': 0.46, 'clip_max': 0.69},
        'Y_Col3': {'base': 0.52, 'trend': 0.008, 'noise': 0.04, 'clip_min': 0.46, 'clip_max': 0.69},
        'Y_Col4': {'base': 0.20, 'trend': 0.010, 'noise': 0.02, 'clip_min': 0.16, 'clip_max': 0.39}
    }
    y_col_config2 = { # Slightly different data for the second set
        'Y_Col1': {'base': 0.52, 'trend': 0.008, 'noise': 0.025, 'clip_min': 0.46, 'clip_max': 0.69},
        'Y_Col2': {'base': 0.53, 'trend': -0.007, 'noise': 0.035, 'clip_min': 0.46, 'clip_max': 0.69},
        'Y_Col3': {'base': 0.50, 'trend': 0.012, 'noise': 0.03, 'clip_min': 0.46, 'clip_max': 0.69},
        'Y_Col4': {'base': 0.22, 'trend': 0.008, 'noise': 0.025, 'clip_min': 0.16, 'clip_max': 0.39}
    }

    if not os.path.exists(dummy_csv_file1):
        create_dummy_csv(dummy_csv_file1, y_col_config=y_col_config1, base_seed_offset=0)
    if not os.path.exists(dummy_csv_file2):
        create_dummy_csv(dummy_csv_file2, y_col_config=y_col_config2, base_seed_offset=100) # Use offset for different data

    # Plot using the dummy data files
    plot_dual_scatter_subplots(dummy_csv_file1, dummy_csv_file2, output_filename="my_dual_plots_1x4_en.png")
    
    # When you have your own CSV files, use the lines below and ensure filenames are correct:
    # plot_dual_scatter_subplots('your_data1.csv', 'your_data2.csv', output_filename='your_actual_plot_name.png')
