import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil

def visualize_combined_sections(start_section=None, middle_section=None, finish_section=None, combined_season=None, save_plots=False):
    """
    Generate all visualizations for the given combined sections (start, middle, finish) or the entire season.
    
    Parameters:
    start_section (DataFrame, optional): The combined start section of player data.
    middle_section (DataFrame, optional): The combined middle section of player data.
    finish_section (DataFrame, optional): The combined finish section of player data.
    combined_season (DataFrame, optional): The combined season data of player data.
    save_plots (bool, optional): Whether to save the generated plots to disk.
    """
    # Create or clear the output directory
    output_dir = 'data_exploration_plots'
    if save_plots:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    if combined_season is not None:
        # Visualizations for the entire combined season
        plot_average_performance(combined_season, 'Entire Season', save_plots, output_dir)
        compare_player_contributions(combined_season, 'Entire Season', save_plots, output_dir)
        plot_correlation_heatmap(combined_season, 'Entire Season', save_plots, output_dir)
    else:
        # Visualizations for each section
        if start_section is not None and middle_section is not None and finish_section is not None:
            # Plot average performance for each section
            plot_average_performance(start_section, 'Start', save_plots, output_dir)
            plot_average_performance(middle_section, 'Middle', save_plots, output_dir)
            plot_average_performance(finish_section, 'Finish', save_plots, output_dir)
            
            # Compare player contributions for each section
            compare_player_contributions(start_section, 'Start', save_plots, output_dir)
            compare_player_contributions(middle_section, 'Middle', save_plots, output_dir)
            compare_player_contributions(finish_section, 'Finish', save_plots, output_dir)
            
            # Plot the performance trend across sections
            plot_performance_trend(start_section, middle_section, finish_section, save_plots, output_dir)
            
            # Plot correlation heatmap for each section
            plot_correlation_heatmap(start_section, 'Start', save_plots, output_dir)
            plot_correlation_heatmap(middle_section, 'Middle', save_plots, output_dir)
            plot_correlation_heatmap(finish_section, 'Finish', save_plots, output_dir)
            
            # Plot player contributions over different seasons
            compare_player_contributions_by_season(start_section, middle_section, finish_section, save_plots, output_dir)
            
            # Plot performance trend by distinguishing different seasons
            plot_performance_trend_by_season(start_section, middle_section, finish_section, save_plots, output_dir)

def plot_average_performance(combined_section, section_name, save_plots, output_dir):
    """
    Plot the average performance metrics (PTS, TRB, AST, BLK, STL) for the given combined section.
    
    Parameters:
    combined_section (DataFrame): The combined section of player data (start, middle, or finish).
    section_name (str): The name of the section being visualized (e.g., 'Start', 'Middle', 'Finish').
    save_plots (bool): Whether to save the plot to disk.
    output_dir (str): Directory to save the plot if save_plots is True.
    """
    # Calculate the average of each metric in the combined section
    average_performance = combined_section[['PTS', 'TRB', 'AST', 'BLK', 'STL']].mean()
    
    # Plot the average performance
    average_performance.plot(kind='bar', title=f'Average Performance Metrics - {section_name} Section')
    plt.xlabel('Metric')
    plt.ylabel('Average Value')
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    ax = plt.gca()
    ax.set_facecolor('white')
    if save_plots:
        plt.savefig(os.path.join(output_dir, f'average_performance_{section_name.lower().replace(" ", "_")}.png'))
    plt.show()

def compare_player_contributions(combined_section, section_name, save_plots, output_dir):
    """
    Compare player contributions across key metrics (PTS, TRB, AST, BLK, STL) in the given combined section.
    
    Parameters:
    combined_section (DataFrame): The combined section of player data (start, middle, or finish).
    section_name (str): The name of the section being visualized (e.g., 'Start', 'Middle', 'Finish').
    save_plots (bool): Whether to save the plot to disk.
    output_dir (str): Directory to save the plot if save_plots is True.
    """
    # Group by player and calculate the mean of each metric (average per game contribution)
    player_contributions = combined_section.groupby('Player')[['PTS', 'TRB', 'AST', 'BLK', 'STL']].mean()
    
    # Plot the contributions of each player
    ax = player_contributions.plot(kind='bar', stacked=True, title=f'Player Contributions - {section_name} Section (Average per Game)', figsize=(10, 6))
    plt.axhline(0, color='black', linewidth=1, linestyle='--')  # Add a line to distinguish where 0 on the y-axis is
    plt.xlabel('Player')
    plt.ylabel('Average Contribution per Game')
    plt.xticks(rotation=90)
    plt.tight_layout()
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    ax = plt.gca()
    ax.set_facecolor('white')
    if save_plots:
        plt.savefig(os.path.join(output_dir, f'player_contributions_{section_name.lower().replace(" ", "_")}.png'))
    plt.show()

def compare_player_contributions_by_season(start_section, middle_section, finish_section, save_plots, output_dir):
    """
    Compare player contributions across key metrics for different seasons (start, middle, finish).
    
    Parameters:
    start_section (DataFrame): The combined start section of player data.
    middle_section (DataFrame): The combined middle section of player data.
    finish_section (DataFrame): The combined finish section of player data.
    save_plots (bool): Whether to save the plot to disk.
    output_dir (str): Directory to save the plot if save_plots is True.
    """
    # Add a column to identify the section (start, middle, finish)
    start_section['Season Section'] = 'Start'
    middle_section['Season Section'] = 'Middle'
    finish_section['Season Section'] = 'Finish'
    
    # Concatenate all sections
    combined_data = pd.concat([start_section, middle_section, finish_section])
    
    # Group by player and season section, and calculate the mean of each metric (average per game contribution)
    player_contributions = combined_data.groupby(['Player', 'Season Section'])[['PTS', 'TRB', 'AST', 'BLK', 'STL']].mean().unstack()
    
    # Define colors for different metrics and sections
    colors = {
        ('PTS', 'Start'): 'lightcoral', ('PTS', 'Middle'): 'red', ('PTS', 'Finish'): 'darkred',
        ('AST', 'Start'): 'lightgreen', ('AST', 'Middle'): 'green', ('AST', 'Finish'): 'darkgreen',
        ('TRB', 'Start'): 'lightblue', ('TRB', 'Middle'): 'blue', ('TRB', 'Finish'): 'darkblue',
        ('BLK', 'Start'): 'lightgray', ('BLK', 'Middle'): 'gray', ('BLK', 'Finish'): 'black',
        ('STL', 'Start'): 'lightyellow', ('STL', 'Middle'): 'yellow', ('STL', 'Finish'): 'gold'
    }
    
    # Plot player contributions across different seasons
    ax = player_contributions.plot(kind='bar', stacked=True, figsize=(15, 8), title='Player Contributions by Season Section (Average per Game)', color=[colors[col] for col in player_contributions.columns])
    plt.axhline(0, color='black', linewidth=1, linestyle='--')  # Add a line to distinguish where 0 on the y-axis is
    plt.xlabel('Player')
    plt.ylabel('Average Contribution per Game')
    plt.xticks(rotation=90)
    
    # Calculate and add the sum of all contributions to the player labels
    player_sums = player_contributions.sum(axis=1)
    labels = [f'{player} ({sum_value:.2f})' for player, sum_value in zip(player_contributions.index, player_sums)]
    ax.set_xticklabels(labels, rotation=90)
    
    plt.tight_layout()
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    ax = plt.gca()
    ax.set_facecolor('white')
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'player_contributions_by_season.png'))
    plt.show()

def plot_performance_trend(start_section, middle_section, finish_section, save_plots, output_dir):
    """
    Plot the trend of average performance metrics (PTS, TRB, AST, BLK, STL) across the start, middle, and finish sections.
    
    Parameters:
    start_section (DataFrame): The combined start section of player data.
    middle_section (DataFrame): The combined middle section of player data.
    finish_section (DataFrame): The combined finish section of player data.
    save_plots (bool): Whether to save the plot to disk.
    output_dir (str): Directory to save the plot if save_plots is True.
    """
    # Calculate the average metrics for each section
    avg_start = start_section[['PTS', 'TRB', 'AST', 'BLK', 'STL']].mean()
    avg_middle = middle_section[['PTS', 'TRB', 'AST', 'BLK', 'STL']].mean()
    avg_finish = finish_section[['PTS', 'TRB', 'AST', 'BLK', 'STL']].mean()
    
    # Create a DataFrame to hold the average values for each section
    trend_data = pd.DataFrame({
        'Start': avg_start,
        'Middle': avg_middle,
        'Finish': avg_finish
    })
    
    # Transpose the DataFrame for plotting
    trend_data = trend_data.T
    
    # Plot the trend of average performance metrics
    trend_data.plot(kind='line', marker='o', title='Performance Trend Across Sections')
    plt.xlabel('Season Section')
    plt.ylabel('Average Value')
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    ax = plt.gca()
    ax.set_facecolor('white')
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'performance_trend_across_sections.png'))
    plt.show()

def plot_performance_trend_by_season(start_section, middle_section, finish_section, save_plots, output_dir):
    """
    Plot the trend of average performance metrics for each season (start, middle, finish).
    
    Parameters:
    start_section (DataFrame): The combined start section of player data.
    middle_section (DataFrame): The combined middle section of player data.
    finish_section (DataFrame): The combined finish section of player data.
    save_plots (bool): Whether to save the plot to disk.
    output_dir (str): Directory to save the plot if save_plots is True.
    """
    # Add a column to identify the section (start, middle, finish)
    start_section['Season Section'] = 'Start'
    middle_section['Season Section'] = 'Middle'
    finish_section['Season Section'] = 'Finish'
    
    # Concatenate all sections
    combined_data = pd.concat([start_section, middle_section, finish_section])
    
    # Group by season section and calculate the average for each metric
    trend_data = combined_data.groupby('Season Section')[['PTS', 'TRB', 'AST', 'BLK', 'STL']].mean()
    
    # Plot the trend of average performance metrics for each season section
    trend_data.plot(kind='line', marker='o', title='Performance Trend by Season Section', figsize=(10, 6))
    plt.xlabel('Season Section')
    plt.ylabel('Average Value')
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    ax = plt.gca()
    ax.set_facecolor('white')
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'performance_trend_by_season_section.png'))
    plt.show()

def plot_correlation_heatmap(combined_section, section_name, save_plots, output_dir):
    """
    Plot a heatmap showing correlations between key metrics (PTS, TRB, AST, BLK, STL) in the given combined section.
    
    Parameters:
    combined_section (DataFrame): The combined section of player data (start, middle, or finish).
    section_name (str): The name of the section being visualized (e.g., 'Start', 'Middle', 'Finish').
    save_plots (bool): Whether to save the plot to disk.
    output_dir (str): Directory to save the plot if save_plots is True.
    """
    # Calculate the correlation matrix for the key metrics
    correlation_matrix = combined_section[['PTS', 'TRB', 'AST', 'BLK', 'STL']].corr()
    
    # Plot the heatmap of the correlation matrix
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title(f'Correlation Heatmap - {section_name} Section')
    plt.tight_layout()
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    ax = plt.gca()
    ax.set_facecolor('white')
    if save_plots:
        plt.savefig(os.path.join(output_dir, f'correlation_heatmap_{section_name.lower().replace(" ", "_")}.png'))
    plt.show()
