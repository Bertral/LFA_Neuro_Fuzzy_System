from sklearn import datasets
from neurofis.nfs import NFS

nfs = NFS(min_observations_per_rule=10)
dataset, target = datasets.make_classification(
    n_samples=500, n_features=6, n_informative=5, n_redundant=0, n_classes=5)
nfs.train(dataset, target, 100, 0.001)
nfs.pruning(dataset)
nfs.test(dataset, target)
nfs.inspect()
