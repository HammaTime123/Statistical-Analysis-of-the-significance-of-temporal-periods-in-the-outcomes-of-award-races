# NBA MVP Award Analysis Project

## Overview

This project aims to analyze the statistical contributions of different periods of an NBA season (start, middle, and finish) to the outcome of the Most Valuable Player (MVP) award race. The analysis combines advanced data processing, visualization, and modeling to determine the relative importance of various performance metrics across these timeframes.

The project uses data sourced from Basketball Reference, normalized against season means and averages. It employs an ensemble model to calculate the importance of various statistics in determining MVP outcomes, based on historical data spanning from 2001 to 2023.

---

## Key Features

### 1. Data Processing
- Aggregates player statistics from raw datasets.
- Splits NBA seasons into sections: start, middle, and finish.
- Supports customizable control over dataset creation and visualization.

### 2. Visualization
- Generates plots and heatmaps to visualize statistical contributions over time and by season section.
- Compares player performance trends across overlapping windows of multiple seasons.

### 3. Statistical Modeling
- Uses sliding window analysis to evaluate the MVP share importance of specific periods.
- Employs regression models to assess and predict MVP outcomes based on key performance metrics.

### 4. Customizable Analysis
- Allows users to decide whether to split the season into sections or analyze the entire season as a whole.
- Provides tools to visualize selected data windows interactively.

### 5. Modular Design
- Well-structured Python modules for data generation, processing, exploration, modeling, and final visualization.

---

## Dependencies

- **Python**: 3.7+
- **Libraries**:
  - `os`
  - `kagglehub`
  - `IPython`
  - Custom modules included in this repository:
    - `DataGeneration.py`
    - `DataProcessing.py`
    - `DataExploration.py`
    - `DataModelling.py`
    - `DataFinalDisplay.py`

---

## Project Structure

- **`DataGeneration.py`**: Functions for collecting and organizing player and MVP statistics.
- **`DataProcessing.py`**: Functions to generate and handle sliding windows of seasonal data.
- **`DataExploration.py`**: Tools for creating visualizations of statistical trends across seasons.
- **`DataModelling.py`**: Functions for analyzing the statistical importance of season sections in MVP predictions.
- **`DataFinalDisplay.py`**: Functions for displaying final trends and MVP importance across overlapping time windows.

---

## How to Use

### 1. Data Preparation
- Ensure raw datasets are placed in the `Raw_player_data_sets` directory.
- Use the provided functions to fetch and preprocess player statistics.

### 2. Visualization
- Generate visualizations for specific season windows or overall trends.
- Choose between splitting the season into sections or using the full season.

### 3. Modeling
- Analyze the statistical importance of different season periods using sliding window data.
- Evaluate MVP share importance over time and across overlapping windows.

### 4. Customizable Control
- Enable or disable interactive control for data exploration and visualization by modifying the `custom_control` parameter.

### 5. Results
- Generated visualizations and trends can be found in the output directories for further analysis.

---

## Outputs

### Visualizations
- Correlation heatmaps for start, middle, and finish sections of the season.
- Player contributions by season and section.
- Overall performance trends across sections.

### Statistical Insights
- Relative importance of various statistics in determining MVP outcomes.
- Trends in MVP share importance over time.

---

## Future Work

- Incorporate analysis of other NBA awards (e.g., Defensive Player of the Year, Rookie of the Year).
- Expand the model to include additional contextual variables, such as team success and playoff impact.
- Enhance the visualization toolkit for more interactive analysis.

---

## License

This project is open-source and available under the MIT License. Contributions are welcome!
