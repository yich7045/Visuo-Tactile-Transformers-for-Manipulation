import argparse
import os
from datetime import datetime
import torch
import minitouch.env
import gym
from slac.push_insert_algo import pushing_SlacAlgorithm
from slac.picking_algo import picking_SlacAlgorithm
from slac.push_insert_trainer import pushing_Trainer
from slac.picking_trainer import picking_Trainer
from slac.open_algo import open_SlacAlgorithm
from slac.open_trainer import open_Trainer
from slac.env import make_dmc

"""
Original Code:
- Toshiki Watanabe, Jan Schneider
- Oct 5, 2021
- slac.pytorch
- 1.6.0
- source code
- https://github.com/ku2482/slac.pytorch
"""

def main(args):
    log_dir = os.path.join(
        "logs",
        f'slac-seed{args.seed}-{datetime.now().strftime("%Y%m%d-%H%M")}',
    )
    if args.task_name == "Inserting-v0" or args.task_name == "InsertingDebug-v0":
        env = make_dmc(args.task_name)
        algo = pushing_SlacAlgorithm()
        trainer = pushing_Trainer(
            env=env,
            algo=algo,
            log_dir=log_dir,
            seed=args.seed,
            num_steps=args.num_steps,
            initial_collection_steps=10000,
            initial_learning_steps=10000
        )
        trainer.train()
    elif args.task_name == "Pushing-v0" or args.task_name == "PushingDebug-v0":
        env = make_dmc(args.task_name)
        algo = pushing_SlacAlgorithm()
        trainer = pushing_Trainer(
            env=env,
            algo=algo,
            log_dir=log_dir,
            seed=args.seed,
            num_steps=args.num_steps,
            initial_collection_steps=50000,
            initial_learning_steps=50000
        )
        trainer.train()
    elif args.task_name == "Picking-v0" or args.task_name == "PickingDebug-v0":
        env = make_dmc(args.task_name)
        algo = picking_SlacAlgorithm()
        trainer = picking_Trainer(
            env=env,
            algo=algo,
            log_dir=log_dir,
            seed=args.seed,
            num_steps=args.num_steps,
            initial_collection_steps=10000,
            initial_learning_steps=10000
        )
        trainer.train()
    else:
        env = make_dmc(args.task_name)
        algo = open_SlacAlgorithm()
        trainer = open_Trainer(
            env=env,
            algo=algo,
            log_dir=log_dir,
            seed=args.seed,
            num_steps=args.num_steps,
            initial_collection_steps=10000,
            initial_learning_steps=10000
        )
        trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=6 * 10 ** 5)
    parser.add_argument("--task_name", type=str, default="Pushing-v0")
    parser.add_argument("--action_repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--encoder", type=str, default="VTT")
    args = parser.parse_args()
    main(args)
