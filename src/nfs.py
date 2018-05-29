import numpy as np
from src.fuzzy_systems.core.rules.fuzzy_rule import FuzzyRule
from sklearn import datasets


class NFS():
    def __init__(self, max_rules: int = 10, min_observations_per_rule: int = 5):
        self._max_rules = max_rules
        self._min_observations_per_rule = min_observations_per_rule
        self._rules = None

    def train(self, data: np.ndarray, target: np.ndarray, nb_iter: int, learning_rate : float):
        """
        Train the NFS for nb_iter complete passes on data
        Data must be a 2-dimentionnal numpy array of this format :
        
        value_A_of_1    value_B_of_1    ...     value_X_of_1
        value_A_of_2    value_B_of_2    ...     value_X_of_2
        ...             ...             ...     ...
        value_A_of_n    value_B_of_n    ...     value_X_of_n
        
        """
        self._rules = []

        membership_functions = {}

        # for each feature, initialize memberships functions
        for feat_index in range(0, np.shape(data)[1]):
            min_obs = np.min(data[:, feat_index])
            max_obs = np.max(data[:, feat_index])
            membership_functions[feat_index] = {}

            # split dataset equally based on current feature (make triangles)
            splits = {}
            for n in range(0, 5):
                splits[(min_obs + n*(max_obs - min_obs)/10,
                        min_obs + (n + 1)*(max_obs - min_obs)/10,
                        min_obs + (n + 2)*(max_obs - min_obs)/10)] = {}

            # for each slice, count the number of observations of each class inbetween splits
            for observation in range(0, np.shape(data)[0]):
                for (k_low, k_mid, k_high) in splits.keys():
                    if k_low <= data[observation, feat_index] <= k_high:

                        # if class dict not initialized, create it
                        if target[observation] not in splits[(k_low, k_mid, k_high)]:
                            splits[(k_low, k_mid, k_high)][target[observation]] = 0

                        # increment the count for this class, slice and feature
                        splits[(k_low, k_mid, k_high)][target[observation]] += 1
                        break
            membership_functions[feat_index] = splits

        # membership functions now is membership_functions[feature_number][(p0, p1, p2)][class] = observ. count

        # TODO find the slice intersections that contain more than _min_observations_per_rule of a single class
        # then start learning
        '''
        élimiter les COMBINAISONS de fuzzy sets ayant moins de _min_observations_per_rule observations
        
        pour chaque observation, trouver la règles la plus active, trouver la distance au centre de cette règle sur
        chaque axe, déplacer le centre vers l'observation, idem pour l'extrêmité (un peu plus), ou l'inverse si la classe
        n'est pas bonne par rapport à la règle.
        
        puis élaguer
        '''
        temp = {}
        # supprime les classes minoritaires du décompte de chaque ensemble
        for feat_key, feat in membership_functions.items():
            temp[feat_key] = {}
            for slice_key, slice in feat.items():
                for key, count in slice.items():
                    if count == max(slice.values()):
                        temp[feat_key][slice_key] = (key, count)

        membership_functions = temp

        '''
        second try
        '''
        mfs = []  # list of lists of fuzzy sets

        # compile grid
        for feat_index in range(0, np.shape(data)[1]):
            min_obs = np.min(data[:, feat_index])
            max_obs = np.max(data[:, feat_index])

            # split dataset equally based on current feature (make triangles)
            splits = []
            for n in range(0, 5):
                splits.append((min_obs + n*(max_obs - min_obs)/10,
                                min_obs + (n + 1)*(max_obs - min_obs)/10,
                                min_obs + (n + 2)*(max_obs - min_obs)/10))
            mfs.append(splits)

        rules = []  # list of dictionnaries (key = list of fuzzy sets, value = dominant class)
        '''
        for observation in range(0, np.shape(data)[0]):
            for (k_low, k_mid, k_high) in splits.keys():
                if k_low <= data[observation, feat_index] <= k_high:
        
                    # if class dict not initialized, create it
                    if target[observation] not in splits[(k_low, k_mid, k_high)]:
                        splits[(k_low, k_mid, k_high)][target[observation]] = 0
        
                    # increment the count for this class, slice and feature
                    splits[(k_low, k_mid, k_high)][target[observation]] += 1
                    break
        '''

    def inspect(self):
        "TODO print FIS"


# test script
nfs = NFS()
iris = datasets.load_iris()
nfs.train(iris.data, iris.target, 100, 0.02)
