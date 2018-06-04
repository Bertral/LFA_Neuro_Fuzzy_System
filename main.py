from sklearn import datasets
from neurofis.nfs import NFS

nfs = NFS(min_observations_per_rule=10, max_rules=3)
dataset = datasets.load_iris()
nfs.train(dataset.data, dataset.target, 100, 0.01)
nfs.pruning(dataset.data)
nfs.test(dataset.data, dataset.target)
nfs.inspect()
'''
nfs = NFS(min_observations_per_rule=10)
dataset, target = datasets.make_classification(
    n_samples=200, n_features=4, n_informative=4, n_redundant=0, n_classes=3)
nfs.train(dataset, target, 100, 0.001)
nfs.pruning(dataset)
nfs.test(dataset, target)
nfs.inspect()
'''