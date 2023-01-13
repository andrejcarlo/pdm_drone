# Final Project PDM Group 6 "Obstacle driven RRT algorithm for quad-rotor motion planning"
Group members:
- Andrei-Carlo Papuc, 4772385
- Benjamin Bogenberger, 5845033
- Crina Mihalache, 4827333
- Simon Gebraad, 4840232

## About this repository

Say here what we implemented and what we sourced, what libraries we used, etc.

## Setting up python environemnt

Setup the conda environemnt using
``` bash
conda env create -f environment.yml
conda activate pdm_drone
pip3 install --upgrade pip
cd ..
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones/
pip3 install -e .
```

## Run applications

- Full simulation (Planning & MPC):
``` bash
python3 main_simulation.py
```
- planning applications ((B)RRT, (B)RRT*, (B)iRRT*, PRM)