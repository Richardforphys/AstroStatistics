import subprocess

# Define the paths
venv_path = "/mnt/c/Users/ricca/LAL/bin/activate"
bash_script_path = "/mnt/c/Users/ricca/SIGNALS/Giaco.sh"

# Function to run the bash script with parameters
def run_bash_script_with_params(params):
    try:
        # Build the full command, adding parameters to the script call
        param_str = ' '.join([f"--{key}={value}" for key, value in params.items()])
        bash_cmd = f"bash -i -c \"source {venv_path} && {bash_script_path} {param_str}\""
        
        # Run the command
        subprocess.run(bash_cmd, shell=True, check=True)
        print("Bash script executed successfully with parameters.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error while running bash script: {e}")

# Function to activate the virtual environment and run the bash script without params
def run_bash_script():
    try:
        # Run the bash script with the virtual environment activated
        bash_cmd = f"bash -i -c \"source {venv_path} && {bash_script_path}\""
        
        # Run the command
        subprocess.run(bash_cmd, shell=True, check=True)
        print("Bash script executed successfully.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error while running bash script: {e}")
