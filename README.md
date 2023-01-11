# Final Project PDM Group 6 "A modified RRT algorithm for quad-rotor motion planning"
Group members:
- Andrei-Carlo Papuc, 4772385
- Benjamin Bogenberger, 5845033
- Crina Mihalache, 4827333
- Simon Gebraad, 4840232

## Python environemnt

TODO needs to be tested

Setup the conda environemnt using
``` bash
conda env create -f environment.yml
conda activate pdm
pip3 install --upgrade pip
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones/
pip3 install -e .
```

## Run applications

- Full simulation (Planning & MPC):
``` bash
python3 main_simulation.py
```
- planning applications TODO
