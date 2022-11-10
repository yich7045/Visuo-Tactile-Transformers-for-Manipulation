# VTT
For CoRL 2022 Paper: Visuo-Tactile Transformers for Manipulation

Requirements:<br />
torch==1.9.0<br />
tqdm==4.48.2<br />
pybullet==3.1.8<br />
gym==0.17.2<br />
matplotlib==3.4.3<br />
numpy==1.21.2<br />
pandas==1.1.2<br />

Minitouch Installation:<br />
cd Minitouch<br />
pip install -e .<br />

Available Tasks: Visualization with Debug mode, e.g., "PushingDebug-v0"<br />
Pushing-v0<br />
Opening-v0<br />
Picking-v0<br />
Inserting-v0<br />


Example of Running Code:<br />
VTT:<br />
python train.py --encoder="VTT" --seed=0 --task_name="Pushing-v0"<br />
Baselines:<br />
python train.py --encoder="POE" --seed=1 --task_name="PickingDebug-v0"<br />
python train.py --encoder="Concatenation" --seed=1 --task_name="Opening-v0"<br />

# Read Results with read_pickle.py<br />

# Credits<br />
The code is based on SLAC.pytorch version and Minitouch but heavily modified.
Orignal Codes:

- Toshiki Watanabe, Jan Schneider
- Oct 5, 2021
- slac.pytorch
- 1.6.0
- source code
- https://github.com/ku2482/slac.pytorch

- Sai Rajeswar, Cyril Ibrahim and Daniel Tremblay
- Jan 8, 2021
- Minitouch
- 0.0.1
- source code
- https://github.com/ServiceNow/MiniTouch
