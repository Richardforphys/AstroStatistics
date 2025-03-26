## What will you find here?
 - This folder contains essentially python notebooks made to help visualize gravitational waves and their effects.
 - The assignments are left by prof. Alberto Sesana teaching Astrophysics of Gravitational Waves course @ Unimib
## The solutions are not complete 
 - Since I'm interested in GW detection for future research purposes I will (if I have spear time) keep updating these exercises and making them more complete

## Folder structure

```plaintext
+---AstroGW
|   |   LAL_vs_AN.ipynb  →  comparison between GW150914 signal generated with LALSim and computed with Post-Newtonian equations
|   |   readme.md
|   |   RingMassesGW.ipynb  →  let's see what happens to N masses on a ring as a GW passes through them
|   |
|   +---GW-signals
|   |   |   GW150914.txt  →  simulated GW150914 signal with LALSim
|   |   |   run_params.py  →  parameters used to generate the signal
|   |   |   run_sim.py  →  script to generate the signal with LALSim
|   |   |   simulated.txt  →  another simulated signal
|   |   |
|   |   +---__pycache__
|   |   |   run_sim.cpython-312.pyc