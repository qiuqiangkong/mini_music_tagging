import random
import torch.distributed as dist


class InfiniteSampler:
    def __init__(self, dataset_size):

        self.indexes = list(range(dataset_size))
        random.shuffle(self.indexes)
        
    def __iter__(self):

        pointer = 0

        while True:

            if pointer == len(self.indexes):
                random.shuffle(self.indexes)
                pointer = 0
                
            index = self.indexes[pointer]
            pointer += 1

            yield index


class DistributedInfiniteSampler:
    def __init__(self, dataset_size):

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        self.indexes = list(range(dataset_size))
        random.shuffle(self.indexes)
        
    def __iter__(self):

        pointer = 0

        while True:

            if pointer >= len(self.indexes):
                random.shuffle(self.indexes)
                pointer = 0
                
            index = self.indexes[pointer + self.rank]
            pointer += self.world_size

            yield index