import numpy as np
labels = np.fromfile('data/packed/ultrachat/train.labels.bin', dtype=np.int32)
tokens = np.fromfile('data/packed/ultrachat/train.bin', dtype=np.uint16)
valid_labels = labels[labels != -100]
print(f'Max token ID in labels: {valid_labels.max()}')
print(f'Min token ID in labels: {valid_labels.min()}')
print(f'Max token ID in tokens: {tokens.max()}')
print(f'Vocab size: 8000')
print(f'Out of range labels (>=8000): {(valid_labels >= 8000).sum()}')
print(f'Out of range tokens (>=8000): {(tokens >= 8000).sum()}')