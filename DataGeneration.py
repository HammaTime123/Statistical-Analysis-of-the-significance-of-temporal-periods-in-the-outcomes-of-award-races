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
    columns_to_keep = ["DATE", "PTS", "AST", "BLK", "STL", "TRB"]
    df_selected = df.loc[:, columns_to_keep]

    # Convert PTS, AST, TRB to numeric, coercing errors to NaN
    df_numeric = df_selected[["PTS", "AST", "TRB","BLK", "STL"]].apply(pd.to_numeric, errors="coerce")

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
    # Ensure that all column names are stripped of leading/trailing whitespace
    mvp_dataframe.columns = mvp_dataframe.columns.str.strip()
    
    # Group players by year
    players_by_year = mvp_dataframe.groupby('year')['Player'].apply(list).to_dict()

    for year, players in players_by_year.items():
        # Strip leading/trailing spaces from player names
        stripped_players = [player.strip() for player in players]
        print(year, ": ", players)
        output_file = f"{output_dir}/mvp_stats_{year}.csv"

        # Check if the file already exists
        if os.path.exists(output_file):
            print(f"Data for year {year} already exists at {output_file}. Checking formatting...")
            
            # Load the existing CSV file and check for column formatting
            existing_data = pd.read_csv(output_file)
            existing_data.columns = existing_data.columns.str.strip()  # Ensure columns are stripped
            
            # Save back the corrected file if columns were modified
            existing_data.to_csv(output_file, index=False)
            print(f"Column names reformatted for year {year} if necessary. Skipping fetching...")
            continue  # Skip to the next year if the file already exists
        
        print(f"Fetching stats for year {year} with {len(stripped_players)} players...")
        # Call the collect_player_stats function for each year
        collect_player_stats(stripped_players, year, include_playoffs, output_file)



        
def check_missing_players(players_by_year: dict, output_dir: str):
    missing_players_by_year = {}

    for year, expected_players in players_by_year.items():
        # Strip leading/trailing spaces from each player's name
        expected_players = [player.strip() for player in expected_players]

        # Define the path to the saved CSV file for this year
        output_file = f"{output_dir}/mvp_stats_{year}.csv"

        # Check if the file exists
        if os.path.exists(output_file):
            # Load the saved player stats for this year
            saved_data = pd.read_csv(output_file)

            # Extract the list of players in the saved data
            if 'Player' in saved_data.columns:
                saved_players = saved_data['Player'].tolist()
                # Strip any whitespace from saved player names
                saved_players = [player.strip() for player in saved_players]
            else:
                print(f"Warning: No 'Player' column found in {output_file}.")
                saved_players = []

            # Identify the missing players
            missing_players = [player for player in expected_players if player not in saved_players]

            # If there are any missing players, add them to the result
            if missing_players:
                missing_players_by_year[year] = missing_players
                print(f"Missing players for year {year}: {missing_players}")
        else:
            print(f"File for year {year} does not exist at {output_file}. All players are considered missing.")
            missing_players_by_year[year] = expected_players

    return missing_players_by_year

def print_missing_players(missing_players_by_year: dict):
    for year, missing_players in missing_players_by_year.items():
        # Print year and number of missing players
        print(f"Year {year}: {len(missing_players)} missing player(s)")
        
        # If there are missing players, print their names
        if missing_players:
            print("Missing Players:")
            for player in missing_players:
                print(f" - {player}")
        print("-" * 40)  # Separator for clarity
