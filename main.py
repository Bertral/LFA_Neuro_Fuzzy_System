from sklearn import datasets
from neurofis.nfs import NFS

nfs = NFS(min_observations_per_rule=10)
dataset, target = datasets.make_classification(n_samples=500, n_features=4, n_informative=3, n_redundant=0, n_classes=3)
nfs.train(dataset, target, 100, 0.001)
nfs.pruning(dataset)
nfs.test(dataset, target)
nfs.inspect()
