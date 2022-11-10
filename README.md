# VTT
The major idea of VTT is in slac.network.encoder.VTT. Code needs to be further cleaned up

Requirements:
torch==1.9.0
tqdm==4.48.2
pybullet==3.1.8
gym==0.17.2
matplotlib==3.4.3
numpy==1.21.2
pandas==1.1.2

Minitouch Installation:
cd Minitouch
pip install -e .

Available Tasks: Visualization with Debug mode, e.g., "PushingDebug-v0"
Pushing-v0
Opening-v0
Picking-v0
Inserting-v0


Example of Running Code:
VTT:
python train.py --encoder="VTT" --seed=0 --task_name="Pushing-v0"
Baselines:
python train.py --encoder="POE" --seed=1 --task_name="PickingDebug-v0"
python train.py --encoder="Concatenation" --seed=1 --task_name="Opening-v0"

# Read Results with read_pickle.py

# Credits
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
