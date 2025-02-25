import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """CleanRL's default layer initialization"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class AMPBuilder:
    def __init__(self, **kwargs):
        self.params = None

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        assert self.params is not None, "params is not set"
        net = AMPBuilder.Network(self.params, **kwargs)
        return net

    class Network(nn.Module):
        def __init__(self, params, **kwargs):
            self.is_continuous = True
            actions_num = kwargs.pop("actions_num")
            input_shape = kwargs.pop("input_shape")
            amp_input_shape = kwargs.get("amp_input_shape")

            self.units = params["mlp"]["units"]  # = [2048, 1536, 1024, 1024, 512, 512]
            self.space_config = params["space"]["continuous"]
            self._disc_units = params["disc"]["units"]  # = [1024, 512]

            super().__init__()

            # Hardcoding MLPs for readability
            # NOTE: How the networs get init is differnt from the original implementation
            actor_input_dim = input_shape[0]
            hidden_output_dim = self.units[-1]
            assert hidden_output_dim == self._disc_units[-1]

            ### Actor
            self.actor_mlp = nn.Sequential(
                layer_init(nn.Linear(actor_input_dim, 2048)),
                nn.SiLU(),
                layer_init(nn.Linear(2048, 1536)),
                nn.SiLU(),
                layer_init(nn.Linear(1536, 1024)),
                nn.SiLU(),
                layer_init(nn.Linear(1024, 1024)),
                nn.SiLU(),
                layer_init(nn.Linear(1024, 512)),
                nn.SiLU(),
                layer_init(nn.Linear(512, hidden_output_dim)),
                nn.SiLU(),
            )
            self.mu = nn.Linear(hidden_output_dim, actions_num)
            # self.mu_act = nn.Identity()  # Remove this

            # NOTE: These config are redundant. Make sure they match,
            assert self.space_config["fixed_sigma"] != self.space_config["learn_sigma"]

            # NOTE: PHC uses fixed sigma
            if self.space_config["fixed_sigma"]:
                self.sigma = nn.Parameter(
                    torch.zeros(actions_num, requires_grad=False, dtype=torch.float32),
                    requires_grad=False,
                )
                nn.init.constant_(self.sigma, self.space_config["sigma_init"]["val"])
            else:
                self.sigma = nn.Linear(hidden_output_dim, actions_num)
                nn.init.constant_(self.sigma.weight, self.space_config["sigma_init"]["val"])

            ### Separate Critic
            self.critic_mlp = nn.Sequential(
                layer_init(nn.Linear(actor_input_dim, 2048)),
                # nn.SiLU(),
                nn.ReLU(),
                layer_init(nn.Linear(2048, 1536)),
                # nn.SiLU(),
                nn.ReLU(),
                layer_init(nn.Linear(1536, 1024)),
                # nn.SiLU(),
                nn.ReLU(),
                layer_init(nn.Linear(1024, 1024)),
                # nn.SiLU(),
                nn.ReLU(),
                layer_init(nn.Linear(1024, 512)),
                # nn.SiLU(),
                nn.ReLU(),
                layer_init(nn.Linear(512, hidden_output_dim)),
                # nn.SiLU(),
                nn.ReLU(),
            )
            self.value = layer_init(nn.Linear(512, 1), std=0.01)
            # self.value = nn.Linear(hidden_output_dim, 1)

            ### Discriminator
            self._disc_mlp = nn.Sequential(
                layer_init(nn.Linear(amp_input_shape[0], 1024)),
                nn.ReLU(),
                layer_init(nn.Linear(1024, hidden_output_dim)),
                nn.ReLU(),
            )

            # self._disc_logits = layer_init(torch.nn.Linear(hidden_output_dim, 1), std=DISC_LOGIT_INIT_SCALE)
            self._disc_logits = layer_init(torch.nn.Linear(hidden_output_dim, 1))

        def forward(self, obs_dict):
            # CHECK ME: obs_dict or obs_dict["obs"] ???
            mu, logstd = self.eval_actor(obs_dict)
            value = self.eval_critic(obs_dict)

            return mu, logstd, value, None

        def eval_actor(self, obs_dict):
            a_out = self.actor_mlp(obs_dict["obs"])
            mu = self.mu(a_out)

            if self.space_config["fixed_sigma"]:
                sigma = self.sigma
            else:
                sigma = self.sigma(a_out)

            return mu, sigma

        def eval_critic(self, obs_dict):
            c_out = self.critic_mlp(obs_dict["obs"])
            return self.value(c_out)

        def eval_disc(self, amp_obs):
            disc_mlp_out = self._disc_mlp(amp_obs)
            disc_logits = self._disc_logits(disc_mlp_out)
            return disc_logits

        def get_disc_logit_weights(self):
            return torch.flatten(self._disc_logits.weight)

        def get_disc_weights(self):
            weights = []
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))

            weights.append(torch.flatten(self._disc_logits.weight))
            return weights


class ModelAMPContinuous:
    def __init__(self, network_builder):
        self.network_builder = network_builder

    def build(self, config):
        net = self.network_builder.build(None, **config)
        # for name, _ in net.named_parameters():
        #     print(name)
        return ModelAMPContinuous.Network(net)

    class Network(nn.Module):
        def __init__(self, a2c_network):
            nn.Module.__init__(self)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None

        def forward(self, input_dict):
            is_train = input_dict.get("is_train", True)
            prev_actions = input_dict.get("prev_actions", None)
            mu, logstd, value, _ = self.a2c_network(input_dict)

            # xcxc debug -- forward (both)
            # print()
            # if prev_actions is not None:
            #     print("prev_actions", prev_actions.sum())
            #     print("mu", mu.sum())
            # print("mu", mu.sum())
            # print("logstd", logstd.sum())
            # print("value", value.sum())
            # print()

            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma)

            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    "prev_neglogp": torch.squeeze(prev_neglogp),
                    "values": value,
                    "entropy": entropy,
                    "rnn_states": None,  # No RNN
                    "mus": mu,
                    "sigmas": sigma,
                }

                ### AMP
                disc_agent_logit = self.a2c_network.eval_disc(input_dict["amp_obs"])
                result["disc_agent_logit"] = disc_agent_logit

                disc_agent_replay_logit = self.a2c_network.eval_disc(input_dict["amp_obs_replay"])
                result["disc_agent_replay_logit"] = disc_agent_replay_logit

                disc_demo_logit = self.a2c_network.eval_disc(input_dict["amp_obs_demo"])
                result["disc_demo_logit"] = disc_demo_logit

                return result

            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    "neglogpacs": torch.squeeze(neglogp),
                    "values": value,
                    "actions": selected_action,
                    "rnn_states": None,  # No RNN
                    "mus": mu,
                    "sigmas": sigma,
                }
                return result

        def neglogp(self, x, mean, std, logstd):
            return (
                0.5 * (((x - mean) / std) ** 2).sum(dim=-1)
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1]
                + logstd.sum(dim=-1)
            )
