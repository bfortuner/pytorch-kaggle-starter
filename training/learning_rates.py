import math
import operator
import copy


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']



class LearningRate():
    def __init__(self, initial_lr, iteration_type):
        self.initial_lr = initial_lr
        self.iteration_type = iteration_type #epoch or mini_batch

    def get_learning_rate(self, optimizer):
        return optimizer.param_groups[0]['lr']

    def set_learning_rate(self, optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    def adjust(self, optimizer, lr, iteration, params=None):
        self.set_learning_rate(optimizer, lr)
        return lr


class FixedLR(LearningRate):
    def __init__(self, initial_lr, iteration_type):
        super().__init__(initial_lr, iteration_type)

    def adjust(self, optimizer, iteration, params=None):
        new_lr = super().get_learning_rate(optimizer)
        return new_lr


class LinearLR(LearningRate):
    def __init__(self, initial_lr, iteration_type, fixed_delta):
        super().__init__(initial_lr, iteration_type)
        self.fixed_delta = fixed_delta

    def adjust(self, optimizer, iteration, params=None):
        lr = super().get_learning_rate(optimizer)
        new_lr = lr + self.fixed_delta
        super().set_learning_rate(optimizer, new_lr)
        return new_lr


class SnapshotLR(LearningRate):
    '''https://arxiv.org/abs/1704.00109'''
    def __init__(self, initial_lr, iteration_type,
                 max_lr, total_iters, n_cycles):
        '''
        n_iters = total number of mini-batch iterations during training
        n_cycles = total num snapshots during training
        max_lr = starting learning rate each cycle'''
        super().__init__(initial_lr, iteration_type)
        self.max_lr = max_lr
        self.total_iters = total_iters
        self.cycles = n_cycles

    def cosine_annealing(self, t):
        '''t = current mini-batch iteration'''
        return self.max_lr/2 * (math.cos(
         (math.pi * (t % (self.total_iters//self.cycles))) /
         (self.total_iters//self.cycles)) + 1)

    def adjust(self, optimizer, iteration, params=None):
        new_lr = self.cosine_annealing(iteration)
        self.set_learning_rate(optimizer, new_lr)
        return new_lr


class SnapshotParamsLR(LearningRate):
    '''Snapshot Learning with per-parameter LRs'''
    def __init__(self, initial_lr, iteration_type,
                 total_iters, n_cycles):
        '''
        n_iters = total number of mini-batch iterations during training
        n_cycles = total num snapshots during training
        max_lr = starting learning rate each cycle'''
        super().__init__(initial_lr, iteration_type)
        self.total_iters = total_iters
        self.cycles = n_cycles

    def cosine_annealing(self, t, max_lr):
        return max_lr/2 * (math.cos(
         (math.pi * (t % (self.total_iters//self.cycles)))/(
            self.total_iters//self.cycles)) + 1)

    def adjust(self, optimizer, iteration, params=None):
        lrs = []
        for param_group in optimizer.param_groups:
            new_lr = self.cosine_annealing(iteration, param_group['max_lr'])
            param_group['lr'] = new_lr
            lrs.append(new_lr)
        return new_lr


class DevDecayLR(LearningRate):
    '''https://arxiv.org/abs/1705.08292'''
    def __init__(self, initial_lr, iteration_type,
                 decay_factor=0.9, decay_patience=1):
        super().__init__(initial_lr, iteration_type)
        self.decay_factor = decay_factor
        self.decay_patience = decay_patience

    def adjust(self, optimizer, iteration, params):
        lr = super().get_learning_rate(optimizer)
        best_iter = params['best_iter']

        if (iteration - best_iter) > self.decay_patience:
            print('Decaying learning rate by factor: {:.5f}'.format(
                self.decay_factor).rstrip('0'))
            lr *= self.decay_factor
            super().set_learning_rate(optimizer, lr)
        return lr


class ScheduledLR(LearningRate):
    def __init__(self, initial_lr, iteration_type, lr_schedule):
        super().__init__(initial_lr, iteration_type)
        self.lr_schedule = lr_schedule

    def adjust(self, optimizer, iteration, params=None):
        if iteration in self.lr_schedule:
            new_lr = self.lr_schedule[iteration]
        else:
            new_lr = self.get_learning_rate(optimizer)
        super().set_learning_rate(optimizer, new_lr)
        return new_lr


class DecayingLR(LearningRate):
    def __init__(self, initial_lr, iteration_type, decay, n_epochs):
         super().__init__(initial_lr, iteration_type)
         self.decay = decay
         self.n_epochs = n_epochs

    def exponential_decay(self, iteration, params=None):
        '''Update learning rate to `initial_lr` decayed
        by `decay` every `n_epochs`'''
        return self.initial_lr * (self.decay ** (iteration // self.n_epochs))

    def adjust(self, optimizer, iteration):
        new_lr = self.exponential_decay(iteration)
        super().set_learning_rate(optimizer, new_lr)
        return new_lr


class CyclicalLR(LearningRate):
    '''https://arxiv.org/abs/1506.01186'''
    def __init__(self, initial_lr, iteration_type, n_iters, cycle_length,
                 min_lr, max_lr):
         assert initial_lr == min_lr
         super(CyclicalLR, self).__init__(initial_lr, iteration_type)
         self.n_iters = n_iters
         self.cycle_length = cycle_length
         self.min_lr = min_lr
         self.max_lr = max_lr

    def triangular(self, iteration):
        iteration -= 1 # if iteration count starts at 1
        cycle = math.floor(1 + iteration/self.cycle_length)
        x = abs(iteration/(self.cycle_length/2) - 2*cycle + 1)
        new_lr = self.min_lr + (self.max_lr - self.min_lr) * max(0, (1-x))
        return new_lr

    def adjust(self, optimizer, iteration, best_iter=1):
        new_lr = self.triangular(iteration)
        super().set_learning_rate(optimizer, new_lr)
        return new_lr




## Helpers

def cosine_annealing(lr_max, T, M, t):
    '''
    t = current mini-batch iteration
    # lr(t) = f(t-1 % T//M)
    # lr(t) = lr_max/2 * (math.cos( (math.pi * (t % T/M))/(T/M) ) + 1)
    '''
    return lr_max/2 * (math.cos( (math.pi * (t % (T//M)))/(T//M)) + 1)
