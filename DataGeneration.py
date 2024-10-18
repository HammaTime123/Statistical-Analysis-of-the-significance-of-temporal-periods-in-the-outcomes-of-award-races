# DataGeneration.py
import os
from basketball_reference_scraper.players import get_stats, get_game_logs
from basketball_reference_scraper.teams import (
    get_roster,
    get_team_stats,
    get_opp_stats,
    get_roster_stats,
    get_team_misc,
)
import pandas as pd
from IPython.display import clear_output
import tqdm
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler



def get_season_mvp():
    file_path_1 = 'MVP_data-sets/2001-2010 MVP Data.csv'
    file_path_2 = 'MVP_data-sets/2010-2021 MVP Data.csv'
    file_path_3 = 'MVP_data-sets/2022-2023 MVP Data.csv'
    temp = load_and_combine_datasets_cleaned(file_path_1, file_path_2, file_path_3)
    return temp



def clear_jupyter():
    clear_output(wait=True)



def get_player_season(player_name, season_year: int, include_playoffs: bool, position: int):
    # Scrape for data
    df = pd.DataFrame(get_game_logs(player_name, season_year, playoffs=include_playoffs))
    clear_jupyter()

    # Choose the columns you are interested in
    columns_to_keep = ["DATE", "PTS", "AST", "TRB"]
    df_selected = df.loc[:, columns_to_keep]

    # Convert PTS, AST, TRB to numeric, coercing errors to NaN
    df_numeric = df_selected[["PTS", "AST", "TRB"]].apply(pd.to_numeric, errors="coerce")

    # Create a mask where all PTS, AST, TRB are not NaN (i.e., are numeric)
    mask = df_numeric.notnull().all(axis=1)

    # Apply the mask to filter out rows with non-numeric values
    df_cleaned = df_selected[mask].reset_index(drop=True)

    # Add the player's name and position using .loc
    df_cleaned.loc[:, "Player"] = player_name
    df_cleaned.loc[:, "Position"] = position

    return df_cleaned



def collect_player_stats(players: list, target_year: int, include_playoffs: bool, output_file: str):
    # ANSI escape codes for green color
    GREEN = "\033[92m"
    RESET = "\033[0m"

    # Define a custom bar format with green color for the progress bar
    custom_bar_format = (
        "{l_bar}"
        f"{GREEN}"  # Start green color
        "{bar}"
        f"{RESET}"  # Reset color
        "| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )

    # Initialize a list to store individual player DataFrames
    mvp_stats_list = []

    # Iterate over each player with a green progress bar
    for intTicker, player in tqdm.tqdm(
        enumerate(players, start=1),
        total=len(players),
        desc="Fetching Player Stats",
        bar_format=custom_bar_format,
    ):
        try:
            # Fetch the player's season statistics
            player_stats = get_player_season(player, target_year, include_playoffs, intTicker)
            # Append the player's stats to the list
            mvp_stats_list.append(player_stats)

            # Log the successful fetch
            logging.info(f"Successfully fetched stats for {player}.")

        except Exception as e:
            # Log any errors encountered
            logging.error(f"Error fetching stats for {player}: {e}")
            continue  # Skip to the next player in case of an error

    # Concatenate all player stats into a single DataFrame
    if mvp_stats_list:
        try:
            mvp_stats = pd.concat(mvp_stats_list, ignore_index=True)
            # Save the combined DataFrame to a CSV file
            if(output_file):
                mvp_stats.to_csv(output_file, index=False)
                logging.info(f"Combined statistics saved to {output_file}.")
        except Exception as e:
            logging.error(f"Error saving DataFrame to CSV: {e}")
            mvp_stats = pd.DataFrame()  # Return an empty DataFrame in case of failure
    else:
        mvp_stats = pd.DataFrame()
        logging.warning("No player stats were fetched. Returning an empty DataFrame.")

    return mvp_stats



def normalize_player_stats(df, method='min-max'):
    # ANSI escape codes for blue color
    RESET = "\033[0m"
    GREEN = "\033[92m"
    
    # Define a custom bar format with blue color for the progress bar
    custom_bar_format = (
        "{l_bar}"
        f"{GREEN}"  # Start blue color
        "{bar}"
        f"{RESET}"  # Reset color
        " | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )
    
    # Define the columns to normalize
    columns_to_normalize = ['PTS', 'AST', 'TRB']
    
    # Check if the necessary columns exist in the DataFrame
    for col in columns_to_normalize:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
    
    # Choose the normalization method
    if method == 'min-max':
        scaler = MinMaxScaler()
    elif method == 'z-score':
        scaler = StandardScaler()
    elif method == 'max-abs':
        scaler = MaxAbsScaler()
    else:
        raise ValueError("Unsupported normalization method. Choose 'min-max', 'z-score', or 'max-abs'.")
    
    # Iterate over each column with a blue progress bar
    for col in tqdm(
        columns_to_normalize, 
        desc="Normalizing Columns", 
        unit="column",
        bar_format=custom_bar_format
    ):
        # Reshape the data for the scaler and overwrite the column with normalized values
        df[col] = scaler.fit_transform(df[[col]])
    
    return df



def load_and_combine_datasets_cleaned(*file_paths):
    datasets = []
    
    for file_path in file_paths:
        # Load each dataset
        data = pd.read_csv(file_path)
        
        # Clean column names by stripping extra spaces
        data.columns = data.columns.str.strip()
        
        # Define columns to keep, adjusted to cleaned names
        columns_to_keep = ["Player", "Rank", "Pts Won", "Pts Max", "Share", "year"]
        
        # Ensure the necessary columns exist before selecting them
        available_columns = [col for col in columns_to_keep if col in data.columns]
        
        # Keep only the desired columns
        cleaned_data = data[available_columns]
        
        datasets.append(cleaned_data)
    
    # Combine the datasets vertically
    combined_data = pd.concat(datasets, axis=0, ignore_index=True)
    
    path = "MVP_data-sets/combined_MVP_data_set.csv"
    combined_data.to_csv(path, index=False)
    return combined_data



def load_combined_mvp_2001_2023():
    path = "MVP_data-sets/combined_MVP_data_set.csv"
    return pd.read_csv(path)




# Function to fetch data for each year using the players of that year
def fetch_mvp_stats_by_year(mvp_dataframe: pd.DataFrame, include_playoffs: bool, output_dir: str):
    players_by_year = mvp_dataframe.groupby('year')['Player'].apply(list).to_dict()

    for year, players in players_by_year.items():
        stripped_players = [player.strip() for player in players]
        print(year, ": ", players)
        output_file = f"{output_dir}/mvp_stats_{year}.csv"

        # Check if the file already exists
        if os.path.exists(output_file):
            print(f"Data for year {year} already exists at {output_file}. Skipping fetching...")
            continue  # Skip to the next year if the file already exists
        
        
        print(f"Fetching stats for year {year} with {len(stripped_players)} players...")
        # Call the collect_player_stats function for each year
        collect_player_stats(stripped_players, year, include_playoffs, output_file)


        