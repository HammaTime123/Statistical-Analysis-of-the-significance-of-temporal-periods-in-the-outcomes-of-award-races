# First, ensure that any previous installations or upgrades are reflected in the current kernel
# by using the `pip` magic command.

def upgrade_pip():
    """
    Upgrades pip to the latest version to ensure compatibility and access to the latest packages.
    """
    try:
        print("Upgrading pip to the latest version...")
        # Use the pip magic command to upgrade pip
        !{sys.executable} -m pip install --upgrade pip
        print("Successfully upgraded pip.\n")
    except Exception as upgrade_error:
        print(f"Failed to upgrade pip: {upgrade_error}")
        print("Continuing with existing pip version.\n")

def install_packages(package_list):
    """
    Attempts to install the specified packages using pip.
    If pip fails, it retries using pip3.

    Parameters:
        package_list (list): A list of package names to install.
    """
    try:
        print(f"Attempting to install packages {package_list} using pip...")
        # Use the pip magic command to install packages
        !{sys.executable} -m pip install {" ".join(package_list)}
        print(f"Successfully installed packages using pip.\n")
    except Exception as pip_error:
        print(f"pip installation failed: {pip_error}")
        print(f"Attempting to install packages {package_list} using pip3...")
        try:
            # Attempt to install using pip3
            !pip3 install {" ".join(package_list)}
            print(f"Successfully installed packages using pip3.\n")
        except Exception as pip3_error:
            print(f"pip3 installation failed: {pip3_error}")
            print(f"Failed to install packages using both pip and pip3.")
            raise RuntimeError("Package installation failed.") from pip3_error

def verify_imports(import_statements):
    """
    Attempts to import each specified module and function to verify successful installation.

    Parameters:
        import_statements (list): A list of import statements as strings.
    """
    print("Verifying package installations by importing them...")
    for stmt in import_statements:
        try:
            exec(stmt)
            print(f"Successfully executed: {stmt}")
        except ImportError as import_error:
            print(f"Failed to execute '{stmt}': {import_error}")
            raise ImportError(f"Import failed for statement: {stmt}") from import_error
        except Exception as e:
            print(f"An error occurred while executing '{stmt}': {e}")
            raise
    print("All specified imports executed successfully.\n")

def setUpEnvironment():
    # List of packages to install
    packages_to_install = [
        "basketball-reference-scraper",
        "pandas",
        "numpy",
        "scikit-learn",
    ]

    # List of import statements to verify installations
    import_statements = [
        "from basketball_reference_scraper.teams import get_roster, get_team_stats, get_opp_stats, get_roster_stats, get_team_misc",
        "from basketball_reference_scraper.players import get_stats, get_game_logs",
        "import pandas as pd",
        "import sklearn",
        "from datetime import timedelta",
        "from datetime import datetime"
    ]

    # Upgrade pip before attempting installations
    upgrade_pip()

    # Install the required packages
    install_packages(packages_to_install)

    # # Verify installations by executing import statements
    # verify_imports(import_statements)
