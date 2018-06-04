from sklearn import datasets
from neurofis.nfs import NFS

nfs = NFS(min_observations_per_rule=10)
iris = datasets.load_iris()
nfs.train(iris.data, iris.target, 1000, 0.001)
nfs.pruning(iris.data)
nfs.test(iris.data, iris.target)
nfs.inspect()
