from run_sim import run_bash_script_with_params

# Example of calling the function with parameters
params = {
    'a': 'IMRPhenomD_NRTidalv2',
    'domain': 'time',
    'm1': 29.1,
    'm2': 36.2,
    'distance': 410,
    'tidal-lambda1': 0.0,
    'tidal-lambda2': 0.0,
    'f-min': 50
}

run_bash_script_with_params(params)