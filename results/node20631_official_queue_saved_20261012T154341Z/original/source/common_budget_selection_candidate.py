"""Preparation only: validated row permutations and author batch-MSE selection."""
import math
TRAIN_ROWS=1281
TRAIN_BATCH=32
EPOCHS=100
DEV_ROWS=229
DEV_BATCH=128
UPDATES_PER_EPOCH=40
TOTAL_UPDATES=4000

def train_batches(frozen_order):
    order=tuple(frozen_order)
    if len(order)!=TRAIN_ROWS or any(type(i) is not int for i in order) or set(order)!=set(range(TRAIN_ROWS)):
        raise ValueError('Expected complete unique official TRAIN positional permutation')
    used=UPDATES_PER_EPOCH*TRAIN_BATCH
    return tuple(order[i:i+TRAIN_BATCH] for i in range(0,used,TRAIN_BATCH)), order[used:]

def author_dev_batch_mse(batch_losses, batch_counts):
    losses=tuple(float(x) for x in batch_losses)
    if tuple(batch_counts)!=(128,101) or len(losses)!=2:
        raise ValueError('Expected fixed official DEV order and batches128/101')
    if any(not math.isfinite(x) or x<0 for x in losses):
        raise ValueError('Invalid DEV batch MSE')
    return sum(losses)/2

def improves_earliest(best, candidate):
    if not math.isfinite(candidate):
        raise ValueError('Nonfinite candidate')
    return candidate<best
