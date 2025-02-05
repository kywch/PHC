import torch
from torch import nn
from torch.utils.data import Dataset

import numpy as np
import gym


numpy_to_torch_dtype_dict = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
}


class DefaultAlgoObserver:
    def before_init(self, base_name, config, experiment_name):
        pass

    def after_init(self, algo):
        self.algo = algo
        self.game_scores = AverageMeter(1, self.algo.games_to_track).to(self.algo.device)
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        if not infos:
            return
        if not isinstance(infos, dict) and len(infos) > 0 and isinstance(infos[0], dict):
            done_indices = done_indices.cpu()
            for ind in done_indices:
                ind = ind.item()
                if len(infos) <= ind // self.algo.num_agents:
                    continue
                info = infos[ind // self.algo.num_agents]
                game_res = None
                if "battle_won" in info:
                    game_res = info["battle_won"]
                if "scores" in info:
                    game_res = info["scores"]

                if game_res is not None:
                    self.game_scores.update(torch.from_numpy(np.asarray([game_res])).to(self.algo.ppo_device))

    def after_steps(self):
        pass

    def after_clear_stats(self):
        self.game_scores.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.game_scores.current_size > 0 and self.writer is not None:
            mean_scores = self.game_scores.get_mean()
            self.writer.add_scalar("scores/mean", mean_scores, frame)
            self.writer.add_scalar("scores/iter", mean_scores, epoch_num)
            self.writer.add_scalar("scores/time", mean_scores, total_time)


class AverageMeter(nn.Module):
    def __init__(self, in_shape, max_size):
        super().__init__()
        self.max_size = max_size
        self.current_size = 0
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0)
        size = np.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean.fill_(0)

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean.squeeze(0).cpu().numpy()


class DefaultRewardsShaper:
    def __init__(self, scale_value=1, shift_value=0, min_val=-np.inf, max_val=np.inf, is_torch=True):
        self.scale_value = scale_value
        self.shift_value = shift_value
        self.min_val = min_val
        self.max_val = max_val
        self.is_torch = is_torch

    def __call__(self, reward):
        reward = reward + self.shift_value
        reward = reward * self.scale_value

        if self.is_torch:
            reward = torch.clamp(reward, self.min_val, self.max_val)
        else:
            reward = np.clip(reward, self.min_val, self.max_val)
        return reward


# NOTE: norm_only is False by default, so that values are clipped to [-5, 5]
class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05, norm_only=False):
        super().__init__()
        print("RunningMeanStd: ", insize)
        self.insize = insize
        self.epsilon = epsilon
        self.norm_only = norm_only

        self.axis = [0]
        self.mean_size  = insize[0]

        self.register_buffer("running_mean", torch.zeros(insize, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(insize, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

        self._frozen = False

    def freeze(self):
        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    def _update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, unnorm=False):
        current_mean = self.running_mean
        current_var = self.running_var

        if unnorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon) * y + current_mean.float()
        else:
            if self.norm_only:
                y = input / torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
                y = torch.clamp(y, min=-5.0, max=5.0)

        # Update after normalization, so that the values used for training and testing are the same.
        if self.training and not self._frozen:
            mean = input.mean(self.axis)  # along channel axis
            var = input.var(self.axis)
            new_mean, new_var, new_count = self._update_mean_var_count_from_moments(self.running_mean, self.running_var, self.count, mean, var, input.size()[0])
            self.running_mean, self.running_var, self.count = new_mean, new_var, new_count

        return y


class AMPDataset(Dataset):
    def __init__(self, batch_size, minibatch_size, horizon_length):
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.horizon_length = horizon_length
        self.length = self.batch_size // self.minibatch_size

        self.special_names = ["rnn_states"]
        self._idx_buf = torch.randperm(batch_size)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self._get_item(idx)

    def update_values_dict(self, values_dict):
        self.values_dict = values_dict

    def update_mu_sigma(self, mu, sigma):
        raise NotImplementedError()

    def _get_item(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        sample_idx = self._idx_buf[start:end]

        input_dict = {}
        for k, v in self.values_dict.items():
            if k not in self.special_names and v is not None:
                input_dict[k] = v[sample_idx]

        if end >= self.batch_size:
            self._shuffle_idx_buf()

        return input_dict

    def _shuffle_idx_buf(self):
        self._idx_buf[:] = torch.randperm(self.batch_size)
        return

    def _get_item_rnn(self, idx):
        # ZL: I am doubling the get_item_rnn function to in a way also get the sequential data. Pretty hacky. 
        # BPTT, input dict is [batch, seqlen, features]. This function return the sequences that are from the same episide and enviornment in sequentila mannar. Not used at the moment since seq_len is set to 1 for RNN right now. 
        step_size = int(self.minibatch_size/self.horizon_length)
        
        start = idx * step_size
        end = (idx + 1) * step_size
        sample_idx = self._idx_buf[start:end]
        
        input_dict = {}
        
        for k,v in self.values_dict.items():
            if k not in self.special_names and v is not None:
                input_dict[k] = v[sample_idx, :].view(step_size * self.horizon_length, -1).squeeze() # flatten to batch size 
        
        input_dict['old_values'] = input_dict['old_values'][:, None] # ZL Hack: following compute assumes that the old_values is [batch, 1], so has to change this back. Otherwise, the loss will be wrong.
        input_dict['returns'] = input_dict['returns'][:, None] # ZL Hack: following compute assumes that the old_values is [batch, 1], so has to change this back. Otherwise, the loss will be wrong.
        
        if self.values_dict['rnn_states'] is not None:
            input_dict['rnn_states'] = [s[sample_idx, :].view(step_size * self.horizon_length, -1) for s in self.values_dict["rnn_states"]]
        
        if (end >= self.batch_size):
            self._shuffle_idx_buf()
        
        return input_dict

class ExperienceBuffer:
    """
    More generalized than replay buffers.
    Implemented for on-policy algos
    """

    def __init__(self, algo_info, device):
        self.algo_info = algo_info
        self.device = device
        self.is_continuous = True

        self.num_agents = algo_info["num_agents"]
        self.num_actors = algo_info["num_actors"]
        self.horizon_length = algo_info["horizon_length"]
        self.obs_base_shape = (self.horizon_length, self.num_agents * self.num_actors)
        self.state_base_shape = (self.horizon_length, self.num_actors)

        self.action_space = algo_info["action_space"]
        self.actions_shape = (self.action_space.shape[0],)
        self.actions_num = self.action_space.shape[0]

        self.tensor_dict = {}
        self.tensor_dict["obses"] = self._create_tensor_from_space(
            algo_info["observation_space"], self.obs_base_shape
        )

        val_space = gym.spaces.Box(low=0, high=1, shape=(algo_info.get("value_size", 1),))
        self.tensor_dict["rewards"] = self._create_tensor_from_space(val_space, self.obs_base_shape)
        self.tensor_dict["values"] = self._create_tensor_from_space(val_space, self.obs_base_shape)
        self.tensor_dict["neglogpacs"] = self._create_tensor_from_space(
            gym.spaces.Box(low=0, high=1, shape=(), dtype=np.float32), self.obs_base_shape
        )
        self.tensor_dict["dones"] = self._create_tensor_from_space(
            gym.spaces.Box(low=0, high=1, shape=(), dtype=np.uint8), self.obs_base_shape
        )
        self.tensor_dict["actions"] = self._create_tensor_from_space(
            gym.spaces.Box(low=0, high=1, shape=self.actions_shape, dtype=np.float32),
            self.obs_base_shape,
        )
        self.tensor_dict["mus"] = self._create_tensor_from_space(
            gym.spaces.Box(low=0, high=1, shape=self.actions_shape, dtype=np.float32),
            self.obs_base_shape,
        )
        self.tensor_dict["sigmas"] = self._create_tensor_from_space(
            gym.spaces.Box(low=0, high=1, shape=self.actions_shape, dtype=np.float32),
            self.obs_base_shape,
        )

    def _create_tensor_from_space(self, space, base_shape):
        if type(space) is gym.spaces.Box:
            dtype = numpy_to_torch_dtype_dict[space.dtype]
            return torch.zeros(base_shape + space.shape, dtype=dtype, device=self.device)

        raise ValueError(f"Unsupported space type: {type(space)}")

    def update_data(self, name, index, val):
        if type(val) is dict:
            for k, v in val.items():
                self.tensor_dict[name][k][index, :] = v
        else:
            self.tensor_dict[name][index, :] = val

    def get_transformed(self, transform_op):
        res_dict = {}
        for k, v in self.tensor_dict.items():
            if type(v) is dict:
                transformed_dict = {}
                for kd, vd in v.items():
                    transformed_dict[kd] = transform_op(vd)
                res_dict[k] = transformed_dict
            else:
                res_dict[k] = transform_op(v)

        return res_dict

    def get_transformed_list(self, transform_op, tensor_list):
        res_dict = {}
        for k in tensor_list:
            v = self.tensor_dict.get(k)
            if v is None:
                continue
            if type(v) is dict:
                transformed_dict = {}
                for kd, vd in v.items():
                    transformed_dict[kd] = transform_op(vd)
                res_dict[k] = transformed_dict
            else:
                res_dict[k] = transform_op(v)

        return res_dict


class ReplayBuffer():
    def __init__(self, buffer_size, device):
        self._head = 0
        self._total_count = 0
        self._buffer_size = buffer_size
        self._device = device
        self._data_buf = None
        self._sample_idx = torch.randperm(buffer_size)
        self._sample_head = 0

    def reset(self):
        self._head = 0
        self._total_count = 0
        self._reset_sample_idx()

    def get_buffer_size(self):
        return self._buffer_size

    def get_total_count(self):
        return self._total_count

    def store(self, data_dict):
        if (self._data_buf is None):
            self._init_data_buf(data_dict)

        n = next(iter(data_dict.values())).shape[0]
        buffer_size = self.get_buffer_size()
        assert(n <= buffer_size)

        for key, curr_buf in self._data_buf.items():
            curr_n = data_dict[key].shape[0]
            assert(n == curr_n)

            store_n = min(curr_n, buffer_size - self._head)
            curr_buf[self._head:(self._head + store_n)] = data_dict[key][:store_n]    
        
            remainder = n - store_n
            if (remainder > 0):
                curr_buf[0:remainder] = data_dict[key][store_n:]  

        self._head = (self._head + n) % buffer_size
        self._total_count += n

    def sample(self, n):
        total_count = self.get_total_count()
        buffer_size = self.get_buffer_size()

        idx = torch.arange(self._sample_head, self._sample_head + n)
        idx = idx % buffer_size
        rand_idx = self._sample_idx[idx]
        if (total_count < buffer_size):
            rand_idx = rand_idx % self._head

        samples = dict()
        for k, v in self._data_buf.items():
            samples[k] = v[rand_idx]

        self._sample_head += n
        if (self._sample_head >= buffer_size):
            self._reset_sample_idx()

        return samples

    def _reset_sample_idx(self):
        buffer_size = self.get_buffer_size()
        self._sample_idx[:] = torch.randperm(buffer_size)
        self._sample_head = 0

    def _init_data_buf(self, data_dict):
        buffer_size = self.get_buffer_size()
        self._data_buf = dict()

        for k, v in data_dict.items():
            v_shape = v.shape[1:]
            self._data_buf[k] = torch.zeros((buffer_size,) + v_shape, device=self._device)
