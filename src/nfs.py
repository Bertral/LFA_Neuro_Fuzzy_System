import numpy as np
import itertools
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
        Data must be a 2-dimentionnal numpy matrix (each row is an observation, each col a feature)
        
        """
        self._rules = []  # list of dictionnaries (key = list of fuzzy sets, value = dominant class)

        '''
        élimiter les COMBINAISONS de fuzzy sets ayant moins de _min_observations_per_rule observations
        
        pour chaque observation, trouver la règles la plus active, trouver la distance au centre de cette règle sur
        chaque axe, déplacer le centre vers l'observation, idem pour l'extrêmité (un peu plus), ou l'inverse si la classe
        n'est pas bonne par rapport à la règle.
        
        puis élaguer
        '''

        mfs = []  # list of lists of fuzzy sets

        # compile fuzzy sets
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

        # make grid squares
        intersections = list(itertools.product(*mfs))  # list of tuples of fuzzy sets (every case of the grid)

        # for each square, add a rule for the highest class if it has enough observations
        for intersection in intersections:
            classes = {}
            for observ in range(0, np.shape(data)[0]):
                found = True
                for feature in range(0, len(data[observ, :])):
                    if not intersection[feature][0] <= data[observ, feature] <= intersection[feature][2]:
                        found = False
                        break
                if found:
                    if target[observ] not in classes:
                        classes[target[observ]] = 0
                    classes[target[observ]] += 1

            if not classes:
                # empty square, go to next
                continue

            # find the target class for this rule
            nb_of_observations, rule_class = max(zip(classes.values(), classes.keys()))
            if nb_of_observations >= self._min_observations_per_rule:
                # use this rule
                self._rules.append({intersection: rule_class})

        print(self._rules)

    def inspect(self):
        "TODO print FIS"


# test script
nfs = NFS(min_observations_per_rule=5)
iris = datasets.load_iris()
nfs.train(iris.data, iris.target, 100, 0.02)
