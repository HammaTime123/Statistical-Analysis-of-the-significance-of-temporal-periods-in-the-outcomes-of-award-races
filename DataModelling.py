# Welcome to the world of Data Modeling! Let's make some MVP-worthy predictions! 🚀
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

# Helper Function Outline
def analyze_season_sections(window_data_iterator, verbose: bool):
    """
    Analyze the importance of each section (start, middle, finish) of a season across a given three-year window.
    
    Parameters:
    window_data_iterator (iterator): An iterator that yields a copy of the current input data for each iteration.
    
    Returns:
    None: 
    """
    for window_data in window_data_iterator:
        # Extract sections from the input data structure
        window_years = window_data['window_years']
        print(f"working on {window_years}")
        combined_start_section = window_data['combined_start_section']
        combined_middle_section = window_data['combined_middle_section']
        combined_finish_section = window_data['combined_finish_section']

        # remove un-neeeded columns
        combined_start_section_prepped = prepare_data_for_regression(combined_start_section)
        combined_middle_section_prepped = prepare_data_for_regression(combined_middle_section)
        combined_finish_section_prepped = prepare_data_for_regression(combined_finish_section)
        
        # Step 1: Generate initial weights using SVR for each section
        start_weights = generate_initial_weights(combined_start_section_prepped, model_type='svr', C=1.0, epsilon=0.1)
        middle_weights = generate_initial_weights(combined_middle_section_prepped, model_type='svr', C=1.0, epsilon=0.1)
        finish_weights = generate_initial_weights(combined_finish_section_prepped, model_type='svr', C=1.0, epsilon=0.1)

        # Print feature importances if verbose is True
        if verbose:
            # Step 1.1: Print feature importance for each section
            print(f"Window Years: {window_years}\n")
            
            # Start Section Feature Importance
            print("Start Section Feature Importance:")
            for feature, importance in start_weights.items():
                if isinstance(importance, (float, int)):
                    print(f"  {feature}: {importance:.4f}")
                else:
                    print(f"  {feature}: {importance}")  # Handle unexpected types

            # Middle Section Feature Importance
            print("\nMiddle Section Feature Importance:")
            for feature, importance in middle_weights.items():
                if isinstance(importance, (float, int)):
                    print(f"  {feature}: {importance:.4f}")
                else:
                    print(f"  {feature}: {importance}")  # Handle unexpected types

            # Finish Section Feature Importance
            print("\nFinish Section Feature Importance:")
            for feature, importance in finish_weights.items():
                if isinstance(importance, (float, int)):
                    print(f"  {feature}: {importance:.4f}")
                else:
                    print(f"  {feature}: {importance}")  # Handle unexpected types
        
        # Step 2: Cross-check weights with tree-based model feature importance for consistency
        start_tree_importance = cross_check_with_tree_importance(combined_start_section_prepped)
        middle_tree_importance = cross_check_with_tree_importance(combined_middle_section_prepped)
        finish_tree_importance = cross_check_with_tree_importance(combined_finish_section_prepped)

        if verbose:
            print("Tree-based Feature Importance - Start Section:")
            for feature, importance in start_tree_importance.items():
                print(f"  {feature}: {importance}")
            print("Tree-based Feature Importance - Middle Section:")
            for feature, importance in middle_tree_importance.items():
                print(f"  {feature}: {importance}")
            print("Tree-based Feature Importance - Finish Section:")
            for feature, importance in finish_tree_importance.items():
                print(f"  {feature}: {importance}")


        mode_start = start_weights['performance']['r2_score'] >= start_tree_importance['performance']['r2_score']
        mode_middle = middle_weights['performance']['r2_score'] >= middle_tree_importance['performance']['r2_score']
        mode_finish = finish_weights['performance']['r2_score'] >= finish_tree_importance['performance']['r2_score']

        print(f"\nmodes: Start: {'SVR' if mode_start else 'Tree Regressor'} -- Middle: {'SVR' if mode_middle else 'Tree Regressor'} -- End: {'SVR' if mode_finish else 'Tree Regressor'}\n")

        # Apply scaling to the start section
        scale_features(
            feature_importance=start_weights['feature_importance'],
            tree_importance=start_tree_importance['feature_importance'],
            mode=mode_start,
            dataframe=combined_start_section,
            section_name='Start'
        )

        # Similarly, apply scaling to the middle and finish sections
        scale_features(
            feature_importance=middle_weights['feature_importance'],
            tree_importance=middle_tree_importance['feature_importance'],
            mode=mode_middle,
            dataframe=combined_middle_section,
            section_name='Middle'
        )

        scale_features(
            feature_importance=finish_weights['feature_importance'],
            tree_importance=finish_tree_importance['feature_importance'],
            mode=mode_finish,
            dataframe=combined_finish_section,
            section_name='Finish'
        )


        aggregated_start = sum_basketball_stats(aggregate_player_stats(combined_start_section))
        aggregated_middle = sum_basketball_stats(aggregate_player_stats(combined_middle_section))
        aggregated_finish = sum_basketball_stats(aggregate_player_stats(combined_finish_section))


        # Combine the aggregated sections into a single DataFrame
        combined_sections = combine_section_stats(aggregated_start, aggregated_middle, aggregated_finish)

        yield combined_sections

# Function to generate initial weights using SVR
def generate_initial_weights(section_data: pd.DataFrame, model_type='svr', C=1.0, epsilon=0.1, test_size=0.2, random_state=42):
    """
    Generate initial weights for statistical categories using Support Vector Regression (SVR) with train-test split.

    Parameters:
    section_data (DataFrame): The input data for a particular section (start, middle, finish).
    model_type (str): The type of regression model to use ('svr').
    C (float): Regularization parameter for the SVR model.
    epsilon (float): Epsilon parameter for the SVR model.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    dict: A dictionary containing:
        - 'feature_importance': Feature importance dictionary.
        - 'performance': Dictionary with 'r2_score' and 'mean_squared_error'.
    """
    # Prepare features and target
    features = section_data.drop(columns=['Share'])
    target = section_data['Share']

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    
    # Select the SVR model
    if model_type == 'svr':
        model = SVR(C=C, epsilon=epsilon)
    else:
        raise ValueError("model_type must be 'svr'")
    
    # Fit the model on training data
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)


    # Permutation Feature Importance
    # Note: The model expects scaled features, so we pass the scaled X_test
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=30,
        random_state=random_state,
        scoring='r2'
    )

    
    # Extract and organize feature importance
    feature_importance = dict(zip(features.columns, result.importances_mean))

    
    return {
        'feature_importance': feature_importance,
        'performance': {
            'r2_score': r2,
            'mean_squared_error': mse
        }
    }

# Function to cross-check weights with tree-based model feature importance
def cross_check_with_tree_importance(section_data: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Cross-check the feature importance using a tree-based model (Random Forest) with train-test split.

    Parameters:
    section_data (DataFrame): The input data for a particular section (start, middle, finish).
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    dict: A dictionary containing:
        - 'feature_importance': Feature importance dictionary from Random Forest.
        - 'performance': Dictionary with 'r2_score' and 'mean_squared_error'.
    """
    # Prepare features and target
    features = section_data.drop(columns=['Share'])
    target = section_data['Share']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    
    # Train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Extract feature importance
    feature_importance = model.feature_importances_
    feature_importance_dict = dict(zip(features.columns, feature_importance))
    
    return {
        'feature_importance': feature_importance_dict,
        'performance': {
            'r2_score': r2,
            'mean_squared_error': mse
        }
    }


def prepare_data_for_regression(total_data: pd.DataFrame):
    """
    Prepare data for regression by dropping unnecessary columns.
    
    Parameters:
    total_data (DataFrame): A DataFrame containing player data.
    
    Returns:
    DataFrame: A DataFrame ready for regression analysis.
    """
    # Drop the specified columns that are not needed for regression
    columns_to_drop = ["DATE", "Position", "Player", "year"]
    regression_data = total_data.drop(columns=columns_to_drop, errors='ignore')
    
    return regression_data

# Function to scale features based on mode
def scale_features(feature_importance, tree_importance, mode, dataframe, section_name):
    """
    Scales the DataFrame's feature columns based on the model mode.

    Parameters:
    - feature_importance (dict): Feature importances from SVR.
    - tree_importance (dict): Feature importances from Tree Regressor.
    - mode (bool): If True, use SVR importance; else use Tree Regressor importance.
    - dataframe (pd.DataFrame): The DataFrame to scale.
    - section_name (str): Name of the section (for logging purposes).

    Returns:
    - None: Modifies the dataframe in place.
    """
    for feature in feature_importance:
        if feature not in dataframe.columns:
            print(f"Warning: Feature '{feature}' not found in the DataFrame '{section_name}'. Skipping scaling for this feature.")
            continue
        scaling_factor = feature_importance[feature] if mode else tree_importance.get(feature, 1.0)
        dataframe[feature] = dataframe[feature] * scaling_factor

import pandas as pd

def aggregate_player_stats(scaled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates player statistics by taking the mean of specified stats per player per year.

    Parameters:
    - scaled_df (pd.DataFrame): The input DataFrame containing player statistics.

    Returns:
    - pd.DataFrame: The aggregated DataFrame with mean statistics per player per year.
    """
    # Define the required columns
    required_columns = ['DATE', 'PTS', 'AST', 'BLK', 'STL', 'TRB', 'Player', 'Position', 'Share', 'year']
    stats_columns = ['PTS', 'AST', 'BLK', 'STL', 'TRB']

    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in scaled_df.columns]
    if missing_columns:
        raise ValueError(f"The following required columns are missing from the input DataFrame: {missing_columns}")

    # Drop 'DATE' and 'Position' columns
    df_dropped = scaled_df.drop(columns=['DATE', 'Position'])

    # Ensure that the stats columns are of float type
    for col in stats_columns:
        if not pd.api.types.is_float_dtype(df_dropped[col]):
            raise TypeError(f"The column '{col}' must be of float type.")

    # Ensure that 'Share' is of float type
    if not pd.api.types.is_float_dtype(df_dropped['Share']):
        raise TypeError("The column 'Share' must be of float type.")

    # Define aggregation functions
    aggregation_functions = {stat: 'mean' for stat in stats_columns}
    aggregation_functions['Share'] = 'mean'  # Adjust as needed

    # Group by 'year' and 'Player' and aggregate using mean
    aggregated_df = df_dropped.groupby(['year', 'Player'], as_index=False).agg(aggregation_functions)

    return aggregated_df


def sum_basketball_stats(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sums specified basketball statistics and retains 'Player' and 'Share' columns.

    Parameters:
    - aggregated_df (pd.DataFrame): The aggregated DataFrame from `aggregate_player_stats`.

    Returns:
    - pd.DataFrame: The DataFrame with 'Player', 'Share', and 'Total_Basketball_Stats'.
    """
    # Define the required columns
    required_columns = ['year', 'Player', 'PTS', 'AST', 'BLK', 'STL', 'TRB', 'Share']
    stat_columns = ['PTS', 'AST', 'BLK', 'STL', 'TRB']

    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in aggregated_df.columns]
    if missing_columns:
        raise ValueError(f"The following required columns are missing from the input DataFrame: {missing_columns}")

    # Ensure that the statistics columns are of numeric type
    for col in stat_columns:
        if not pd.api.types.is_numeric_dtype(aggregated_df[col]):
            raise TypeError(f"The column '{col}' must be of a numeric type (int or float).")

    # Ensure that 'Share' is of numeric type
    if not pd.api.types.is_numeric_dtype(aggregated_df['Share']):
        raise TypeError("The column 'Share' must be of a numeric type (int or float).")

    # Sum the specified basketball statistics for each row
    aggregated_df['Total_Basketball_Stats'] = aggregated_df[stat_columns].sum(axis=1)

    # Retain only 'Player', 'Share', and 'Total_Basketball_Stats'
    final_df = aggregated_df[['Player', "year",'Share', 'Total_Basketball_Stats']]

    return final_df




def combine_section_stats(aggregated_start: pd.DataFrame, 
                         aggregated_middle: pd.DataFrame, 
                         aggregated_finish: pd.DataFrame) -> pd.DataFrame:
    """
    Combine aggregated player statistics from start, middle, and finish sections into a single DataFrame.
    Identify and report any inconsistencies in 'Share' values.
    
    Parameters:
    - aggregated_start (pd.DataFrame): Aggregated stats for the start section with columns ['Player', 'year', 'Share', 'Total_Basketball_Stats'].
    - aggregated_middle (pd.DataFrame): Aggregated stats for the middle section with columns ['Player', 'year', 'Share', 'Total_Basketball_Stats'].
    - aggregated_finish (pd.DataFrame): Aggregated stats for the finish section with columns ['Player', 'year', 'Share', 'Total_Basketball_Stats'].
    
    Returns:
    - pd.DataFrame: Combined DataFrame with columns ['Player', 'year', 'Share', 
                                                    'Total_Basketball_Stats_start', 
                                                    'Total_Basketball_Stats_middle', 
                                                    'Total_Basketball_Stats_finish'].
    """
    # Define the required columns
    required_columns = {'Player', 'year', 'Share', 'Total_Basketball_Stats'}
    
    # Verify that required columns are present in each DataFrame
    for section_name, df in zip(['Start', 'Middle', 'Finish'], 
                                 [aggregated_start, aggregated_middle, aggregated_finish]):
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing columns in {section_name} section DataFrame: {missing}")

    # Rename 'Total_Basketball_Stats' to include section name
    start_renamed = aggregated_start.rename(columns={'Total_Basketball_Stats': 'Total_Basketball_Stats_start'})
    middle_renamed = aggregated_middle.rename(columns={'Total_Basketball_Stats': 'Total_Basketball_Stats_middle'})
    finish_renamed = aggregated_finish.rename(columns={'Total_Basketball_Stats': 'Total_Basketball_Stats_finish'})

    # Merge start and middle sections on 'Player' and 'year'
    merged_start_middle = pd.merge(start_renamed, middle_renamed, on=['Player', 'year'], 
                                   how='outer', suffixes=('_start', '_middle'))

    # Merge the result with finish section
    merged_all = pd.merge(merged_start_middle, finish_renamed, on=['Player', 'year'], 
                          how='outer')

    # Identify section-specific 'Share' columns (exclude the general 'Share' column)
    share_columns = [col for col in merged_all.columns if col.startswith('Share_')]

    if len(share_columns) > 1:
        # Define a tolerance for floating-point comparison to handle minor precision differences
        tolerance = 1e-6
        
        # Check consistency using np.allclose to allow minor differences
        merged_all['Share_consistent'] = merged_all[share_columns].apply(
            lambda row: np.allclose(row, row.iloc[0], atol=tolerance), axis=1
        )
        
        inconsistent_shares = merged_all[~merged_all['Share_consistent']]
        
        if not inconsistent_shares.empty:
            # Extract relevant information about inconsistencies
            inconsistent_info = inconsistent_shares[['Player', 'year'] + share_columns]
            
            # Convert the inconsistent rows to a string for the error message
            inconsistent_details = inconsistent_info.to_string(index=False)
            
            # Raise a ValueError with detailed information
            raise ValueError(
                f"Inconsistent 'Share' values found across sections for the following Player-Year combinations:\n{inconsistent_details}"
            )
        
        # If consistent, assign one of the 'Share' columns to 'Share' and drop the others
        merged_all['Share'] = merged_all['Share_start']
        
        # Drop only the section-specific 'Share' columns and the helper column
        merged_all = merged_all.drop(columns=share_columns + ['Share_consistent'])
    elif len(share_columns) == 1:
        # Only one section-specific 'Share' column exists; assign it to 'Share'
        merged_all['Share'] = merged_all[share_columns[0]]
        
        # Drop the original section-specific 'Share' column
        merged_all = merged_all.drop(columns=share_columns[0])
    else:
        # No section-specific 'Share' columns found; check if 'Share' exists
        if 'Share' in merged_all.columns:
            pass  # 'Share' already exists; no action needed
        else:
            raise ValueError("No 'Share' columns found in the merged DataFrame.")

    # Select and order the desired columns
    desired_columns = ['Player', 'year', 'Share', 
                       'Total_Basketball_Stats_start', 
                       'Total_Basketball_Stats_middle', 
                       'Total_Basketball_Stats_finish']
    
    # Check if all desired columns are present
    missing_desired = set(desired_columns) - set(merged_all.columns)
    if missing_desired:
        raise ValueError(f"Missing expected columns in the combined DataFrame: {missing_desired}")
    
    # Create the final combined DataFrame
    combined_df = merged_all[desired_columns]

    return combined_df

