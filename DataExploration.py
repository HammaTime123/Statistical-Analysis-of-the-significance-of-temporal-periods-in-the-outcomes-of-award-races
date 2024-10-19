import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_combined_sections(start_section, middle_section, finish_section):
    """
    Generate all visualizations for the given combined sections (start, middle, finish).
    
    Parameters:
    start_section (DataFrame): The combined start section of player data.
    middle_section (DataFrame): The combined middle section of player data.
    finish_section (DataFrame): The combined finish section of player data.
    """
    # Plot average performance for each section
    plot_average_performance(start_section, 'Start')
    plot_average_performance(middle_section, 'Middle')
    plot_average_performance(finish_section, 'Finish')
    
    # Compare player contributions for each section
    compare_player_contributions(start_section, 'Start')
    compare_player_contributions(middle_section, 'Middle')
    compare_player_contributions(finish_section, 'Finish')
    
    # Plot the performance trend across sections
    plot_performance_trend(start_section, middle_section, finish_section)
    
    # Plot correlation heatmap for each section
    plot_correlation_heatmap(start_section, 'Start')
    plot_correlation_heatmap(middle_section, 'Middle')
    plot_correlation_heatmap(finish_section, 'Finish')
    
    # Plot player contributions over different seasons
    compare_player_contributions_by_season(start_section, middle_section, finish_section)
    
    # Plot performance trend by distinguishing different seasons
    plot_performance_trend_by_season(start_section, middle_section, finish_section)

def plot_average_performance(combined_section, section_name):
    """
    Plot the average performance metrics (PTS, TRB, AST) for the given combined section.
    
    Parameters:
    combined_section (DataFrame): The combined section of player data (start, middle, or finish).
    section_name (str): The name of the section being visualized (e.g., 'Start', 'Middle', 'Finish').
    """
    # Calculate the average of each metric in the combined section
    average_performance = combined_section[['PTS', 'TRB', 'AST']].mean()
    
    # Plot the average performance
    average_performance.plot(kind='bar', title=f'Average Performance Metrics - {section_name} Section')
    plt.xlabel('Metric')
    plt.ylabel('Average Value')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def compare_player_contributions(combined_section, section_name):
    """
    Compare player contributions across key metrics (PTS, TRB, AST) in the given combined section.
    
    Parameters:
    combined_section (DataFrame): The combined section of player data (start, middle, or finish).
    section_name (str): The name of the section being visualized (e.g., 'Start', 'Middle', 'Finish').
    """
    # Group by player and calculate the sum of each metric
    player_contributions = combined_section.groupby('Player')[['PTS', 'TRB', 'AST']].sum()
    
    # Plot the contributions of each player
    player_contributions.plot(kind='bar', stacked=True, title=f'Player Contributions - {section_name} Section', figsize=(10, 6))
    plt.xlabel('Player')
    plt.ylabel('Total Contribution')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def compare_player_contributions_by_season(start_section, middle_section, finish_section):
    """
    Compare player contributions across key metrics for different seasons (start, middle, finish).
    
    Parameters:
    start_section (DataFrame): The combined start section of player data.
    middle_section (DataFrame): The combined middle section of player data.
    finish_section (DataFrame): The combined finish section of player data.
    """
    # Add a column to identify the section (start, middle, finish)
    start_section['Season Section'] = 'Start'
    middle_section['Season Section'] = 'Middle'
    finish_section['Season Section'] = 'Finish'
    
    # Concatenate all sections
    combined_data = pd.concat([start_section, middle_section, finish_section])
    
    # Group by player and season section, and calculate the sum of each metric
    player_contributions = combined_data.groupby(['Player', 'Season Section'])[['PTS', 'TRB', 'AST']].sum().unstack()
    
    # Plot player contributions across different seasons
    player_contributions.plot(kind='bar', stacked=True, figsize=(15, 8), title='Player Contributions by Season Section')
    plt.xlabel('Player')
    plt.ylabel('Total Contribution')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_performance_trend(start_section, middle_section, finish_section):
    """
    Plot the trend of average performance metrics (PTS, TRB, AST) across the start, middle, and finish sections.
    
    Parameters:
    start_section (DataFrame): The combined start section of player data.
    middle_section (DataFrame): The combined middle section of player data.
    finish_section (DataFrame): The combined finish section of player data.
    """
    # Calculate the average metrics for each section
    avg_start = start_section[['PTS', 'TRB', 'AST']].mean()
    avg_middle = middle_section[['PTS', 'TRB', 'AST']].mean()
    avg_finish = finish_section[['PTS', 'TRB', 'AST']].mean()
    
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
    plt.show()

def plot_performance_trend_by_season(start_section, middle_section, finish_section):
    """
    Plot the trend of average performance metrics for each season (start, middle, finish).
    
    Parameters:
    start_section (DataFrame): The combined start section of player data.
    middle_section (DataFrame): The combined middle section of player data.
    finish_section (DataFrame): The combined finish section of player data.
    """
    # Add a column to identify the section (start, middle, finish)
    start_section['Season Section'] = 'Start'
    middle_section['Season Section'] = 'Middle'
    finish_section['Season Section'] = 'Finish'
    
    # Concatenate all sections
    combined_data = pd.concat([start_section, middle_section, finish_section])
    
    # Group by season section and calculate the average for each metric
    trend_data = combined_data.groupby('Season Section')[['PTS', 'TRB', 'AST']].mean()
    
    # Plot the trend of average performance metrics for each season section
    trend_data.plot(kind='line', marker='o', title='Performance Trend by Season Section', figsize=(10, 6))
    plt.xlabel('Season Section')
    plt.ylabel('Average Value')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(combined_section, section_name):
    """
    Plot a heatmap showing correlations between key metrics (PTS, TRB, AST) in the given combined section.
    
    Parameters:
    combined_section (DataFrame): The combined section of player data (start, middle, or finish).
    section_name (str): The name of the section being visualized (e.g., 'Start', 'Middle', 'Finish').
    """
    # Calculate the correlation matrix for the key metrics
    correlation_matrix = combined_section[['PTS', 'TRB', 'AST']].corr()
    
    # Plot the correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title(f'Correlation Heatmap - {section_name} Section')
    plt.tight_layout()
    plt.show()
