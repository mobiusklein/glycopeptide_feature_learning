import numpy as np


class Histogram(object):
    def __init__(self, data=None, edges=None):
        if edges is None:
            if data is not None:
                _, edges = np.histogram(data, bins='auto')
            else:
                data = []
        self.bins = []
        if edges is not None:
            self.edges = list(edges)
            for edge in edges:
                self.bins.append([])
        else:
            self.bins.append([])
            self.edges = []
        self._size = 0
        if data is not None:
            self.update(data)

    def bin_index(self, value):
        i = 0
        for i, edge in enumerate(self.edges):
            if value < edge:
                break
        return i

    def update(self, data):
        for value in data:
            i = self.bin_index(value)
            self.bins[i].append(value)
            self._size += 1

    def add(self, value):
        self.update([value])

    def __len__(self):
        return self._size

    def __iter__(self):
        return iter(self.bins)

    def __getitem__(self, i):
        return self.bins[i]

    def equalize(self, nbins, data=None):
        if data is None:
            data = []
            tuple(map(data.extend, self))
        data = sorted(data)
        self.bins = []
        for i in range(nbins):
            self.bins.append([])
        self.edges = []
        bin_size = len(data) // nbins
        for i in range(nbins):
            start = i * bin_size
            self.bins[i].extend(data[start:(start + bin_size)])
            self.edges.append(data[start])
            if i == nbins - 1:
                self.bins[i].extend(data[(start + bin_size):])
        self._size = len(data)
