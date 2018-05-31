import numpy as np
import itertools
from src.fuzzy_systems.core.rules.fuzzy_rule import FuzzyRule
from src.mf import MF
from sklearn import datasets

from src.point import Point


class NFS:
    def __init__(self, max_rules: int = 10, min_observations_per_rule: int = 5):
        self._max_rules = max_rules
        self._min_observations_per_rule = min_observations_per_rule
        self._rules = None

    def train(self, data: np.ndarray, target: np.ndarray, nb_iter: int, learning_rate: float):
        """
        Train the NFS for nb_iter complete passes on data
        Data must be a 2-dimentionnal numpy matrix (each row is an observation, each col a feature)
        
        """
        self._rules = {}  # dictionnary (key = tuple of fuzzy sets, value = dominant class)

        """
        élimiter les COMBINAISONS de fuzzy sets ayant moins de _min_observations_per_rule observations
        
        pour chaque observation, trouver la règles la plus active, trouver la distance au centre de cette règle sur
        chaque axe, déplacer le centre vers l'observation, idem pour l'extrêmité (un peu plus), ou l'inverse si la classe
        n'est pas bonne par rapport à la règle.
        
        puis élaguer
        """
        print("Building default fuzyy sets ...")

        mfs = []  # list of lists of fuzzy sets

        # compile fuzzy sets
        for feat_index in range(0, np.shape(data)[1]):
            min_obs = np.min(data[:, feat_index])
            max_obs = np.max(data[:, feat_index])

            # split dataset equally based on current feature (make triangles)
            points = []
            for n in range(0, 7):
                points.append(Point(min_obs + n * (max_obs - min_obs) / 7))
            splits = []
            for n in range(0, 5):
                splits.append(MF(points[n], points[n+1], points[n+2]))
            mfs.append(splits)

        # make grid squares
        intersections = list(itertools.product(*mfs))  # list of tuples of fuzzy sets (every case of the grid)
        print("Rules without consequent built : " + str(len(intersections)))

        print("Finding rule consequents and removing weak rules ...")
        # for each square, add a rule for the highest class if it has enough observations
        for intersection in intersections:
            classes = {}
            for observ in range(0, np.shape(data)[0]):
                found = True
                for feature in range(0, len(data[observ, :])):
                    if not intersection[feature].low.x <= data[observ, feature] <= intersection[feature].high.x:
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
                self._rules[intersection] = rule_class
        print("Rules found : " + str(len(self._rules)))

        print("Repairing holes left by deleted membership functions ...")
        for feature in range(0, np.shape(data)[1]):
            for rule in self._rules.keys():
                # looks for the neighbor of the membership function rule[feature]
                has_neighbour = False
                for other_rule in self._rules.keys():
                    if rule[feature].mid == other_rule[feature].low and rule[feature].high == other_rule[feature].mid:
                        has_neighbour = True
                        break  # neighbour found
                if not has_neighbour:
                    nighbour = None
                    dist = float('+infinity')
                    # find the next nearest membership function
                    for other_rule in self._rules.keys():
                        if rule[feature].high < other_rule[feature].mid and other_rule[feature].mid - rule[feature].high < dist:


    def inspect(self):
        """TODO print FIS"""


# test script
nfs = NFS(min_observations_per_rule=10)
iris = datasets.load_iris()
nfs.train(iris.data, iris.target, 100, 0.02)
