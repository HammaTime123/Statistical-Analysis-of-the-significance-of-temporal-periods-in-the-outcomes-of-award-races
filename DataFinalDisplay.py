import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Ensure matplotlib is imported
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
from typing import List, Any, Union
import logging
import scipy.stats as stats

def analyze_mvp_share_importance_over_time_sliding_windows(
    window_data_iterator,
    window_identifiers: List[Any],
    verbose: bool = True,
    plot_trends: bool = True,
    plot_predictions: Union[bool, List[Any]] = True,  # Modified parameter
    test_size: float = 0.2,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Analyze the relative importance of season sections (start, middle, finish) on the MVP 'Share' across multiple overlapping time windows.
    
    Parameters:
    - window_data_iterator (iterator): An iterator that yields combined_sections DataFrames, one for each sliding window.
    - window_identifiers (List[Any]): List of identifiers (e.g., starting year) for each window.
    - verbose (bool): If True, print detailed outputs.
    - plot_trends (bool): If True, generate plots of feature importance trends over time.
    - plot_predictions (bool or List[Any]): If True, generate scatter plots of predicted vs actual MVP shares for all windows.
      If False, do not generate plots. If a list of window identifiers, generate plots only for those windows.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Seed used by the random number generator.
    
    Returns:
    - feature_importances_over_time (pd.DataFrame): DataFrame containing feature importances across models and windows.
    - stat_results_df (pd.DataFrame): DataFrame containing statistical testing results for feature importance trends.
    """
    
    # Initialize logging
    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(message)s')
    
    logging.info("=== Starting MVP Share Importance Analysis Across Multiple Sliding Windows ===")
    
    # Define feature columns once
    feature_columns = ['Total_Basketball_Stats_start', 
                       'Total_Basketball_Stats_middle', 
                       'Total_Basketball_Stats_finish']
    
    # Initialize a list to store feature importances per window
    all_feature_importances = []
    
    # Determine which windows to plot
    if isinstance(plot_predictions, list):
        plot_windows = plot_predictions
    elif isinstance(plot_predictions, bool):
        if plot_predictions:
            plot_windows = window_identifiers  # Plot for all windows
        else:
            plot_windows = []  # Plot for none
    else:
        raise ValueError("plot_predictions should be either a boolean or a list of window identifiers.")
    
    # Initialize a counter to track the number of processed windows
    processed_windows = 0
    
    # Iterate through each window
    for window_id, combined_sections in zip(window_identifiers, window_data_iterator):
        processed_windows += 1
        logging.info(f"\n=== Analyzing Window {processed_windows}/{len(window_identifiers)}: {window_id} ===")
        
        # Log the shape and columns of the DataFrame
        logging.info(f"DataFrame Shape: {combined_sections.shape}")
        logging.info(f"DataFrame Columns: {combined_sections.columns.tolist()}")
        
        # Check for missing values
        missing_values = combined_sections.isnull().sum()
        if missing_values.any():
            logging.info("Missing values detected. Handling missing values by dropping rows with any missing data.")
            logging.info(missing_values[missing_values > 0])
            combined_sections_clean = combined_sections.dropna()
            logging.info(f"DataFrame Shape after dropping missing values: {combined_sections_clean.shape}")
        else:
            logging.info("No missing values detected.")
            combined_sections_clean = combined_sections.copy()
        
        # Define Features and Target
        target_column = 'Share'
        
        # Ensure all required columns are present
        missing_features = [col for col in feature_columns if col not in combined_sections_clean.columns]
        if missing_features:
            logging.error(f"In the window '{window_id}', the following required feature columns are missing: {missing_features}")
            continue  # Skip this window
        
        if target_column not in combined_sections_clean.columns:
            logging.error(f"In the window '{window_id}', the target column '{target_column}' is missing from the DataFrame.")
            continue  # Skip this window
        
        X = combined_sections_clean[feature_columns]
        y = combined_sections_clean[target_column]
        
        # Check if X or y is empty
        if X.empty or y.empty:
            logging.warning(f"Window '{window_id}' has empty features or target after preprocessing. Skipping.")
            continue  # Skip this window
        
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logging.info(f"Training set size: {X_train.shape[0]} samples")
        logging.info(f"Testing set size: {X_test.shape[0]} samples")
        
        # Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize a dictionary to store importances for this window
        window_importances = {'Window': window_id}
        
        # 1. Lasso Regression
        logging.info("\n--- Lasso Regression ---")
        try:
            lasso = Lasso(alpha=0.1, random_state=random_state)
            lasso.fit(X_train_scaled, y_train)
            y_pred_lasso = lasso.predict(X_test_scaled)
            lasso_r2 = r2_score(y_test, y_pred_lasso)
            lasso_mse = mean_squared_error(y_test, y_pred_lasso)
            lasso_importance = np.abs(lasso.coef_)
            lasso_feature_importance = dict(zip(feature_columns, lasso_importance))
            window_importances['Lasso_R2'] = lasso_r2
            window_importances['Lasso_MSE'] = lasso_mse
            for feature in feature_columns:
                window_importances[f'Lasso_{feature}'] = lasso_feature_importance[feature]
            
            logging.info(f"Lasso R2 Score: {lasso_r2:.4f}")
            logging.info(f"Lasso Mean Squared Error: {lasso_mse:.4f}")
            logging.info("Lasso Coefficients:")
            for feature, coef in zip(feature_columns, lasso.coef_):
                logging.info(f"  {feature}: {coef:.4f}")
            
            # Plotting Predicted vs Actual for Lasso Regression
            if window_id in plot_windows:
                plt.figure(figsize=(8, 6))
                plt.scatter(y_test, y_pred_lasso, alpha=0.7, edgecolors='b')
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                plt.title(f"Lasso Regression Predicted vs Actual MVP Shares (Window {window_id})")
                plt.xlabel("Actual MVP Shares")
                plt.ylabel("Predicted MVP Shares")
                plt.tight_layout()
                plt.show()
        except Exception as e:
            logging.error(f"Lasso Regression failed for window '{window_id}': {e}")
            continue  # Skip to the next window
        
        # 2. Ridge Regression
        logging.info("\n--- Ridge Regression ---")
        try:
            ridge = Ridge(alpha=1.0, random_state=random_state)
            ridge.fit(X_train_scaled, y_train)
            y_pred_ridge = ridge.predict(X_test_scaled)
            ridge_r2 = r2_score(y_test, y_pred_ridge)
            ridge_mse = mean_squared_error(y_test, y_pred_ridge)
            ridge_importance = np.abs(ridge.coef_)
            ridge_feature_importance = dict(zip(feature_columns, ridge_importance))
            window_importances['Ridge_R2'] = ridge_r2
            window_importances['Ridge_MSE'] = ridge_mse
            for feature in feature_columns:
                window_importances[f'Ridge_{feature}'] = ridge_feature_importance[feature]
            
            logging.info(f"Ridge R2 Score: {ridge_r2:.4f}")
            logging.info(f"Ridge Mean Squared Error: {ridge_mse:.4f}")
            logging.info("Ridge Coefficients:")
            for feature, coef in zip(feature_columns, ridge.coef_):
                logging.info(f"  {feature}: {coef:.4f}")
            
            # Plotting Predicted vs Actual for Ridge Regression
            if window_id in plot_windows:
                plt.figure(figsize=(8, 6))
                plt.scatter(y_test, y_pred_ridge, alpha=0.7, edgecolors='g')
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                plt.title(f"Ridge Regression Predicted vs Actual MVP Shares (Window {window_id})")
                plt.xlabel("Actual MVP Shares")
                plt.ylabel("Predicted MVP Shares")
                plt.tight_layout()
                plt.show()
        except Exception as e:
            logging.error(f"Ridge Regression failed for window '{window_id}': {e}")
            continue  # Skip to the next window
        
        # 3. Support Vector Regression (SVR)
        logging.info("\n--- Support Vector Regression (SVR) ---")
        try:
            svr = SVR(C=1.0, epsilon=0.1)
            svr.fit(X_train_scaled, y_train)
            y_pred_svr = svr.predict(X_test_scaled)
            svr_r2 = r2_score(y_test, y_pred_svr)
            svr_mse = mean_squared_error(y_test, y_pred_svr)
            
            # Permutation Importance for SVR
            logging.info("Calculating Permutation Importances for SVR...")
            perm_importance_svr = permutation_importance(
                svr, X_test_scaled, y_test, n_repeats=30, random_state=random_state, scoring='r2'
            )
            svr_feature_importance = dict(zip(feature_columns, perm_importance_svr.importances_mean))
            window_importances['SVR_R2'] = svr_r2
            window_importances['SVR_MSE'] = svr_mse
            for feature in feature_columns:
                window_importances[f'SVR_{feature}'] = svr_feature_importance[feature]
            
            logging.info(f"SVR R2 Score: {svr_r2:.4f}")
            logging.info(f"SVR Mean Squared Error: {svr_mse:.4f}")
            logging.info("SVR Permutation Feature Importances:")
            for feature, importance in svr_feature_importance.items():
                logging.info(f"  {feature}: {importance:.4f}")
            
            # Plotting Predicted vs Actual for SVR
            if window_id in plot_windows:
                plt.figure(figsize=(8, 6))
                plt.scatter(y_test, y_pred_svr, alpha=0.7, edgecolors='m')
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                plt.title(f"SVR Predicted vs Actual MVP Shares (Window {window_id})")
                plt.xlabel("Actual MVP Shares")
                plt.ylabel("Predicted MVP Shares")
                plt.tight_layout()
                plt.show()
        except Exception as e:
            logging.error(f"SVR failed for window '{window_id}': {e}")
            continue  # Skip to the next window
        
        # 4. Random Forest Regressor
        logging.info("\n--- Random Forest Regressor ---")
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            rf_r2 = r2_score(y_test, y_pred_rf)
            rf_mse = mean_squared_error(y_test, y_pred_rf)
            rf_feature_importance = dict(zip(feature_columns, rf.feature_importances_))
            window_importances['RF_R2'] = rf_r2
            window_importances['RF_MSE'] = rf_mse
            for feature in feature_columns:
                window_importances[f'RF_{feature}'] = rf_feature_importance[feature]
            
            logging.info(f"Random Forest R2 Score: {rf_r2:.4f}")
            logging.info(f"Random Forest Mean Squared Error: {rf_mse:.4f}")
            logging.info("Random Forest Feature Importances:")
            for feature, importance in rf_feature_importance.items():
                logging.info(f"  {feature}: {importance:.4f}")
            
            # Plotting Predicted vs Actual for Random Forest
            if window_id in plot_windows:
                plt.figure(figsize=(8, 6))
                plt.scatter(y_test, y_pred_rf, alpha=0.7, edgecolors='c')
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                plt.title(f"Random Forest Predicted vs Actual MVP Shares (Window {window_id})")
                plt.xlabel("Actual MVP Shares")
                plt.ylabel("Predicted MVP Shares")
                plt.tight_layout()
                plt.show()
        except Exception as e:
            logging.error(f"Random Forest Regression failed for window '{window_id}': {e}")
            continue  # Skip to the next window
        
        # Append this window's importances to the list
        all_feature_importances.append(window_importances)
        
        logging.info(f"Window '{window_id}' Analysis Complete.\n")
    
    # After processing all windows
    logging.info("\n=== Aggregating Feature Importances Across All Windows ===")
    feature_importances_over_time = pd.DataFrame(all_feature_importances)
    logging.info("\nFeature Importances DataFrame:")
    logging.info(feature_importances_over_time.head())
    
    # 5. Statistical Testing: Assess if feature importances have significant trends over time
    logging.info("\n=== Statistical Testing of Feature Importances Over Time ===")
    
    # Convert window_identifiers to numerical values for regression (e.g., starting year)
    try:
        window_numeric = [int(wid) for wid in window_identifiers]
    except:
        window_numeric = list(range(len(window_identifiers)))
        logging.warning("Window identifiers are not numerical. Using window index as numeric values for statistical testing.")
    
    stat_results = []
    
    feature_models = ['Lasso', 'Ridge', 'SVR', 'RF']
    
    for feature in feature_columns:
        for model in feature_models:
            importance_col = f'{model}_{feature}'
            if importance_col not in feature_importances_over_time.columns:
                logging.error(f"Column '{importance_col}' not found in feature_importances_over_time DataFrame.")
                continue  # Skip if the column doesn't exist
            X_stat = np.array(window_numeric)
            y_stat = feature_importances_over_time[importance_col].values
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(window_numeric, y_stat)
            stat_results.append({
                'Feature': feature,
                'Model': model,
                'Slope': slope,
                'R_squared': r_value**2,
                'p_value': p_value
            })
    
    stat_results_df = pd.DataFrame(stat_results)
    
    logging.info("\nLinear Regression Results for Feature Importances Over Time:")
    logging.info(stat_results_df)
    
    # 6. Visualization of Feature Importances Over Time
    if plot_trends:
        logging.info("\n=== Plotting Feature Importances Over Time ===")
        
        # Melt the DataFrame for seaborn compatibility
        feature_importances_melt = feature_importances_over_time.melt(id_vars=['Window'], 
                                                                       var_name='Model_Feature', 
                                                                       value_name='Importance')
        
        # Extract Model and Feature from 'Model_Feature'
        feature_importances_melt[['Model', 'Feature']] = feature_importances_melt['Model_Feature'].str.split('_', 
                                                                                                               n=1, 
                                                                                                               expand=True)
        
        # Filter to only include feature importances (exclude R2 and MSE)
        feature_importances_filtered = feature_importances_melt[feature_importances_melt['Feature'].isin(feature_columns)]
        
        # Plot using seaborn's lineplot
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=feature_importances_filtered, x='Window', y='Importance', hue='Feature', style='Model', markers=True)
        plt.title("Feature Importances Over Time Across Season Sections")
        plt.xlabel("Window Identifier (Starting Year)")
        plt.ylabel("Importance Score")
        plt.legend(title='Season Section')
        plt.tight_layout()
        plt.show()
    
    # 7. Summary of Findings
    logging.info("\n=== Summary of Findings ===")
    
    if not feature_importances_over_time.empty:
        # List of models used
        feature_models = ['Lasso', 'Ridge', 'SVR', 'RF']

        # Compute average importance per feature across all models
        for feature in feature_columns:
            model_feature_cols = [f'{model}_{feature}' for model in feature_models]
            # Ensure all columns exist
            existing_cols = [col for col in model_feature_cols if col in feature_importances_over_time.columns]
            if existing_cols:
                feature_importances_over_time[feature] = feature_importances_over_time[existing_cols].mean(axis=1)
            else:
                logging.warning(f"No data available to compute average importance for feature '{feature}'.")

        # Compute average importance across features for each model
        for model in feature_models:
            model_feature_cols = [f'{model}_{feature}' for feature in feature_columns if f'{model}_{feature}' in feature_importances_over_time.columns]
            if model_feature_cols:
                feature_importances_over_time[f'Average_Importance_{model}'] = feature_importances_over_time[model_feature_cols].mean(axis=1)
            else:
                logging.warning(f"No data available to compute average importance for model '{model}'.")

        # Overall average importance across all models
        average_model_importance_cols = [f'Average_Importance_{model}' for model in feature_models if f'Average_Importance_{model}' in feature_importances_over_time.columns]
        if average_model_importance_cols:
            feature_importances_over_time['Overall_Average_Importance'] = feature_importances_over_time[
                average_model_importance_cols
            ].mean(axis=1)
        else:
            logging.warning("No average importances per model found. Cannot compute 'Overall_Average_Importance'.")

        # Identify the most important feature in each window
        feature_importances_over_time['Most_Important_Feature'] = feature_importances_over_time[
            feature_columns].idxmax(axis=1)
        
        logging.info("\nMost Important Features Per Window:")
        logging.info(feature_importances_over_time[['Window', 'Most_Important_Feature']])
        
        # Additionally, provide the overall average importance
        overall_feature_importance = feature_importances_over_time[feature_columns].mean().sort_values(ascending=False)
        logging.info("\nOverall Average Feature Importances Across All Windows:")
        logging.info(overall_feature_importance)
    else:
        logging.warning("No feature importances were recorded. Check if the generator is yielding data correctly.")
    
    return feature_importances_over_time, stat_results_df
