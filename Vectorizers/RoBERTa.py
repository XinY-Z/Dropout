import torch

roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta.eval

