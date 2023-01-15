# Final Project PDM Group 6 "Obstacle driven RRT algorithm for quad-rotor motion planning"
Group members:
- Andrei-Carlo Papuc, 4772385
- Benjamin Bogenberger, 5845033
- Crina Mihalache, 4827333
- Simon Gebraad, 4840232

## About this repository

This repository encapsulates a quad-rotor motion planning stack with various implementations for offline planning ((biased) RRT, (biased) RRT*, (biased) informed RRT*, PRM) and online MPC control. This stack is integrated in a physics simulation with multiple environments. On top of that evaluation scripts and results are provided.

<img src="saved_results_simulation/room_2_72-5percent.gif" alt="room 2 animation" width="700">
<img src="saved_results_simulation/20230114_202129.gif" alt="mpc pybullet demo", width=700>

This repository was developed using Python 3.8. Its dependencies are:
- NumPy
- Matplotlib
- scikit-learn
- CVXPY
- gym-pybullet-drones (v1.0.0)
- tqdm
- ipykernel

The repository is structured as follows.
- `src/` contains the main code
- `main_simulation.py` executes the full planning and control stack in a physics simulation
- `main_rrt.py`, `main_prm.py` and `main_rooms.py` show their respective planning part
- the Jupyter notebooks contain code for evaluation of the planning algorithms
- `saved_results`, `saved_results_hole`, `saved_results_simulation` and `plots` contain main evaluation data
- `linearize_quadrotor_model.m` symbolically linearizes a quad-rotor model

## Setting up python environment

Set up the conda environment using
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
python3 main_simulation.py (--room 0...3) (--planner RRT,RRT_s,iRRT_s,PRM)
```
- Planning applications alone ((B)RRT, (B)RRT*, (B)iRRT*, PRM): `main_rrt.py` and `main_prm.py`
