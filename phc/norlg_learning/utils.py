import torch
from torch import nn
import numpy as np


def shape_whc_to_cwh(shape):
    # if len(shape) == 2:
    #    return (shape[1], shape[0])
    if len(shape) == 3:
        return (shape[2], shape[0], shape[1])

    return shape


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
            import torch

            reward = torch.clamp(reward, self.min_val, self.max_val)
        else:
            reward = np.clip(reward, self.min_val, self.max_val)
        return reward


class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super().__init__()
        print("RunningMeanStd: ", insize)
        self.insize = insize
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0, 2, 3]
            if len(self.insize) == 2:
                self.axis = [0, 2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0]
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer("running_mean", torch.zeros(in_size, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(in_size, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

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

    def forward(self, input, unnorm=False, mask=None):
        if self.training:
            if mask is not None:
                raise NotImplementedError
                # mean, var = torch_ext.get_mean_std_with_masks(input, mask)
            else:
                mean = input.mean(self.axis)  # along channel axis
                var = input.var(self.axis)
            self.running_mean, self.running_var, self.count = self._update_mean_var_count_from_moments(
                self.running_mean, self.running_var, self.count, mean, var, input.size()[0]
            )

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
        return y
