import torch
import torch.nn.utils.rnn as rnn
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
import torchaudio
import os

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

trainset = SubsetSC('training')
labels = sorted(list(set(datapoint[2] for datapoint in trainset)))

def format_data(new_sample_rate, sample_rate, waveform):
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    transformed = transform(waveform)
    return transformed, transform

def label_to_index(word):
    return labels.index(word)

def index_to_label(index):
    return labels[index]

def pad_sequence(batch):
    batch = [item.t() for item in batch]
    batch = rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def collate_fn(batch):
    tensors, targets = [], []
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]  # Convert label to index

    tensors = pad_sequence(tensors)
    targets = torch.tensor(targets)  # Ensure shape [batch_size]
    return tensors, targets

def dataloaders(trainset, testset, batch_size, num_workers, pin_memory):
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return trainloader, testloader
