import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def generate_3_wide_window(players_by_year: dict):
    """
    Generate a list of 3-year windows from the given dictionary of players by year.
    
    Parameters:
    players_by_year (dict): A dictionary where keys are years and values are lists of players.
    
    Returns:
    list: A list of lists, where each inner list contains three consecutive years.
    """
    # Get the sorted list of unique years from the dictionary keys
    years = sorted(players_by_year.keys())
    
    # Create a list of lists where each list is a 3-wide window over the years
    # This helps in analyzing data over a moving window of three years
    windowed_years = [years[i:i + 3] for i in range(len(years) - 2)]
    
    return windowed_years
  

def normalize_columns_auto(dfs):
    """
    Normalize specified columns across a list of dataframes.
    
    Parameters:
    dfs (list of DataFrame): A list of pandas DataFrames to be normalized.
    
    Returns:
    list of DataFrame: A list of normalized DataFrames.
    """
    # Columns to normalize
    columns_to_normalize = ["PTS", "TRB", "AST", "BLK", "STL"]
    
    # Concatenate all the dataframes to calculate the combined mean and standard deviation
    combined_df = pd.concat(dfs)
    
    # Calculate mean and standard deviation for the specified columns
    mean_values = combined_df[columns_to_normalize].mean()
    std_values = combined_df[columns_to_normalize].std()

    # Normalize the specified columns for each dataframe
    normalized_dfs = []
    for df in dfs:
        # Create a copy of the dataframe to avoid modifying the original data
        df_copy = df.copy()
        # Apply normalization: (value - mean) / standard deviation
        df_copy[columns_to_normalize] = (df_copy[columns_to_normalize] - mean_values) / std_values
        normalized_dfs.append(df_copy)
    
    return normalized_dfs

def split_season_into_sections(season_df):
    """
    Split a season dataframe into start, middle, and finish sections.
    
    Parameters:
    season_df (DataFrame): A pandas DataFrame containing season data, including a 'DATE' column.
    
    Returns:
    tuple: A tuple containing three DataFrames: start_section, middle_section, finish_section.
    """
    # Convert the 'DATE' column to datetime format to enable sorting and time-based operations
    season_df['DATE'] = pd.to_datetime(season_df['DATE'])
    # Sort the dataframe by date to ensure the season is in chronological order
    season_df = season_df.sort_values(by='DATE')
    
    # Split the dataframe into three equal parts: start, middle, and finish
    total_games = len(season_df)
    split_size = total_games // 3

    # The start section contains the first third of the games
    start_section = season_df.iloc[:split_size]
    # The middle section contains the second third of the games
    middle_section = season_df.iloc[split_size:2*split_size]
    # The finish section contains the remaining games
    finish_section = season_df.iloc[2*split_size:]
    
    return start_section, middle_section, finish_section

def get_iterable_window_data(players_by_year, data_path: str, mvp_data_path: str, split_into_sections=True):
    """
    Generate an iterator over combined data sections for each 3-year window.
    
    Parameters:
    players_by_year (dict): A dictionary where keys are years and values are lists of players.
    data_path (str): The path to the directory containing the season data CSV files.
    mvp_data_path (str): The path to the MVP CSV data file.
    split_into_sections (bool): Whether to split the season data into start, middle, and finish sections.
    
    Yields:
    dict: A dictionary containing window_years, and combined data sections (split or whole).
    """
    # Generate the 3-year windows using the players_by_year dictionary
    windows = generate_3_wide_window(players_by_year)
    
    # Iterate over each window
    for window in windows:
        # Print the years in the current window
        print(f"Processing window with years: {window}")
        
        # Load the data for each year in the window
        dfs = []
        for year in window:
            file_path = os.path.join(data_path, f"mvp_stats_{year}.csv")
            if os.path.exists(file_path):
                # Read the CSV file into a dataframe
                df = pd.read_csv(file_path)
                # Add 'year' column to match MVP data
                df['year'] = year
                dfs.append(df)
            else:
                # Print a warning if the file for the year is not found
                print(f"Warning: Data file for year {year} not found at {file_path}")
        
        # Normalize the loaded data using normalize_columns_auto
        if dfs:
            normalized_dfs = normalize_columns_auto(dfs)
            
            if split_into_sections:
                # Split each normalized season into start, middle, and finish sections
                start_sections = []
                middle_sections = []
                finish_sections = []
                for df in normalized_dfs:
                    # Split the season into three sections
                    start, middle, finish = split_season_into_sections(df)
                    # Append each section to the respective list
                    start_sections.append(start)
                    middle_sections.append(middle)
                    finish_sections.append(finish)
                
                # Concatenate the respective sections from all seasons in the window
                combined_start_section = pd.concat(start_sections)
                combined_middle_section = pd.concat(middle_sections)
                combined_finish_section = pd.concat(finish_sections)
                
                # Add MVP share to each combined section
                combined_start_section = add_mvp_share_to_sections(combined_start_section, mvp_data_path)
                combined_middle_section = add_mvp_share_to_sections(combined_middle_section, mvp_data_path)
                combined_finish_section = add_mvp_share_to_sections(combined_finish_section, mvp_data_path)
                
                # Yield the combined sections as an iterator
                yield {
                    "window_years": window,
                    "combined_start_section": combined_start_section,
                    "combined_middle_section": combined_middle_section,
                    "combined_finish_section": combined_finish_section
                }
            else:
                # Concatenate the entire seasons from all years in the window
                combined_season = pd.concat(normalized_dfs)
                
                # Add MVP share to the combined season
                combined_season = add_mvp_share_to_sections(combined_season, mvp_data_path)
                
                # Yield the combined season as an iterator
                yield {
                    "window_years": window,
                    "combined_season": combined_season
                }

def print_combined_section_intervals(section_list):
    """
    Helper function to print the three intervals that have been concatenated to produce a combined section.
    
    Parameters:
    section_list (list of DataFrame): The list of individual sections that were concatenated to produce the combined section.
    """
    print("Combined Section Intervals:")
    for idx, section in enumerate(section_list):
        # Get the start date of the interval
        start_date = section['DATE'].iloc[0] if not section.empty else 'N/A'
        # Get the end date of the interval
        end_date = section['DATE'].iloc[-1] if not section.empty else 'N/A'
        # Get the length of the interval
        length = len(section)
        # Print the interval details
        print(f"Interval {idx + 1}: Start Date: {start_date}, End Date: {end_date}, Length: {length}")
        print("---")

def add_mvp_share_to_sections(section_data, mvp_data_path):
    """
    Add MVP share to each player in the provided sections based on the player name and year.
    
    Parameters:
    section_data (DataFrame): A pandas DataFrame containing player statistics for a section.
    mvp_data_path (str): The path to the MVP CSV data file.
    
    Returns:
    DataFrame: The updated section data with MVP share added.
    """
    # Load the MVP data from the CSV file
    mvp_data = pd.read_csv(mvp_data_path)
    
    # Ensure the column names are consistent
    if 'Year' in mvp_data.columns:
        mvp_data.rename(columns={'Year': 'year'}, inplace=True)
    if 'Year' in section_data.columns:
        section_data.rename(columns={'Year': 'year'}, inplace=True)
    
    # Extract the year from the DATE column in section_data if year column is missing
    if 'year' not in section_data.columns:
        section_data['year'] = pd.to_datetime(section_data['DATE']).dt.year
    
    # Make sure column names are consistent for merging
    mvp_data.rename(columns={"Player": "Player", "year": "year"}, inplace=True)
    
    # Print columns to ensure consistency before merging (for debugging)
    print("Section Data Columns:", section_data.columns)
    print("MVP Data Columns:", mvp_data.columns)
    
    # Merge the section data with the MVP data based on 'Player' and 'year'
    try:
        section_data = section_data.merge(mvp_data[['Player', 'year', 'Share']], on=['Player', 'year'], how='left')
    except KeyError as e:
        print(f"KeyError during merging: {e}")
    
    return section_data
