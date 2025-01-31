import os
import ast
import argparse
import configparser

import pufferlib
import pufferlib.vector

from phc.pufferl.demo.environment import make_env
from phc.pufferl.demo.policy import CleanRLPolicy
from phc.pufferl.demo import clean_pufferl

if __name__ == "__main__":
    p = configparser.ConfigParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    p.read(os.path.join(current_dir, "phc.ini"))

    parser = argparse.ArgumentParser()
    for section in p.sections():
        for key in p[section]:
            if section == "base":
                argparse_key = f"--{key}".replace("_", "-")
            else:
                argparse_key = f"--{section}.{key}".replace("_", "-")
            parser.add_argument(argparse_key, default=p[section][key])

    parsed = parser.parse_args().__dict__
    args = {"env": {}, "policy": {}, "rnn": {}}
    for key, value in parsed.items():
        next = args
        for subkey in key.split("."):
            if subkey not in next:
                next[subkey] = {}
            prev = next
            next = next[subkey]
        try:
            prev[subkey] = ast.literal_eval(value)
        except:
            prev[subkey] = value

    vec = pufferlib.environment.PufferEnv  # Native vecenv
    args["env"]["num_envs"] = args["train"]["num_envs"]

    device = args["train"]["device"]

    env_kwargs = {
        "cfg": {
            "env": args["env"],
            "robot": args["robot"],
        },
        "device_type": device,
        "device_id": 0,
        "headless": True,
    }

    vecenv = pufferlib.vector.make(
        make_env,
        env_kwargs=env_kwargs,
        backend=vec,
    )

    policy = CleanRLPolicy(vecenv).to(device)

    train_config = pufferlib.namespace(**args["train"], env="puffer_phc")

    data = clean_pufferl.create(train_config, vecenv, policy)
    while data.global_step < train_config.total_timesteps:
        clean_pufferl.evaluate(data)
        clean_pufferl.train(data)

    uptime = data.profile.uptime
    steps_evaluated = 0
    steps_to_eval = int(args["train"]["eval_timesteps"])
    batch_size = args["train"]["batch_size"]
    while steps_evaluated < steps_to_eval:
        stats, _ = clean_pufferl.evaluate(data)
        steps_evaluated += batch_size

    clean_pufferl.mean_and_log(data)
