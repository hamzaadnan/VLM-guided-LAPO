import math
import torch
import functools

def _linear_decay(iteration, total_iterations):
    return 1.0 - (iteration / total_iterations)

def _linear_decay_warmup(iteration, warmup_iterations, total_iterations):
    if iteration < warmup_iterations:
        multiplier = iteration / warmup_iterations
    else:
        multiplier = 1.0 - ((iteration - warmup_iterations) / (total_iterations - warmup_iterations))
    return multiplier

def _cosine_decay(iteration, total_iterations):
    multiplier = iteration / total_iterations
    multiplier = 0.5 * (1 + math.cos(math.pi * multiplier))
    return multiplier

def _cosine_decay_warmup(iteration, warmup_iterations, total_iterations):
    if iteration < warmup_iterations:
        multiplier = iteration / warmup_iterations
    else:
        multiplier = (iteration - warmup_iterations) / (total_iterations - warmup_iterations)
        multiplier = 0.5 * (1 + math.cos(math.pi * multiplier))
    return multiplier

def _constant_warmup_cooldown(iteration, warmup_iterations, cooldown_iterations, total_iterations):
    if iteration < warmup_iterations:
        multiplier = iteration / warmup_iterations
    elif warmup_iterations <= iteration < (total_iterations - cooldown_iterations):
        multiplier = 1.0
    else:
        multiplier = 1 - math.sqrt((iteration - (total_iterations - cooldown_iterations)) / cooldown_iterations)
    return multiplier

def _constant_warmup(iteration, warmup_iterations):
    multiplier = 1.0
    if iteration < warmup_iterations:
        multiplier = iteration / warmup_iterations
    return multiplier

def cosine_annealing(optimiser, total_steps):
    _decay_func = functools.partial(
        _cosine_decay, total_iterations=total_steps
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimiser, lr_lambda=_decay_func)
    return scheduler

def cosine_annealing_with_warmup(optimiser, warmup_steps, total_steps):
    _decay_func = functools.partial(
        _cosine_decay_warmup, warmup_iterations=warmup_steps, total_iterations=total_steps
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimiser, lr_lambda=_decay_func)
    return scheduler

def linear_warmup(optimiser, warmup_steps):
    _decay_func = functools.partial(
        _constant_warmup, warmup_iterations=warmup_steps
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimiser, lr_lambda=_decay_func)
    return scheduler


def constant_with_cooldown(optimiser, warmup_steps, cooldown_steps, total_steps):
    _decay_func = functools.partial(
        _constant_warmup_cooldown,
        warmup_iterations=warmup_steps,
        cooldown_iterations=cooldown_steps,
        total_iterations=total_steps
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimiser, lr_lambda=_decay_func)
    return scheduler

def linear_annealing(optimiser, total_steps):
    _delay_func = functools.partial(
        _linear_decay, total_iterations=total_steps
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimiser, lr_lambda=_delay_func)
    return scheduler

def linear_annealing_with_warmup(optimiser, warmup_steps, total_steps):
    _decay_func = functools.partial(
        _linear_decay_warmup,
        warmup_iterations=warmup_steps,
        total_iterations=total_steps
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimiser, lr_lambda=_decay_func)
    return scheduler