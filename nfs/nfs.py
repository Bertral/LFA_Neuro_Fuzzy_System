import numpy as np
import itertools
from fuzzy_systems.core.rules.fuzzy_rule import FuzzyRule
from nfs.mf import MF
from sklearn import datasets

from nfs.point import Point


class NFS:
    def __init__(self, max_rules: int = 10, min_observations_per_rule: int = 5):
        self._nb_of_features = 0
        self._max_rules = max_rules
        self._min_observations_per_rule = min_observations_per_rule
        self._rules = None

    def train(self, data: np.ndarray, target: np.ndarray, nb_iter: int, learning_rate: float):
        """
        Train the NFS for nb_iter complete passes on data
        Data must be a 2-dimentionnal numpy matrix (each row is an observation, each col a feature)
        
        """
        self._rules = {}  # dictionnary (key = tuple of fuzzy sets, value = dominant class)
        self._nb_of_features = np.shape(data)[1]

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
                splits.append(MF(points[n], points[n + 1], points[n + 2]))
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

        self.repair(np.shape(data)[1])

        print("Training ...")
        for i in range(0, nb_iter):
            for obs in range(0, np.shape(data)[0]):
                # find the most activated rule for this observation
                max_rule = None
                max_act = 0
                for mfs, target_class in self._rules.items():
                    act = 0
                    # activate
                    for feat in range(0, len(mfs)):
                        act += mfs[feat].fuzzyfy(data[obs, feat]) / len(mfs)
                    # compare activation with max_act
                    if act > max_act:
                        max_rule = (mfs, target_class)
                        max_act = act
                if max_rule is None:
                    continue  # skip if the observation has no rule
                # adjust membership functions for the most activated rule baseed on this observation
                for feat in range(0, len(max_rule[0])):
                    # move membership function to/away from (if same/different class) data[obs, feat] on distance
                    # learning_rate
                    max_rule[0][feat].move(data[obs, feat], learning_rate, max_rule[1] == target[obs])

        print("Training done !")

    def repair(self, nb_of_features):
        """
        Repairs holes in linguistic variables if membership functions were deleted
        """
        print("Repairing holes left by deleted membership functions ...")
        for feature in range(0, nb_of_features):
            for rule in self._rules.keys():
                # looks for the nearest neighbor of the membership function "rule[feature]"
                neighbour = None
                dist = float('+infinity')
                # find the next nearest membership function
                for other_rule in self._rules.keys():
                    if rule[feature].high.x < other_rule[feature].mid.x \
                            and other_rule[feature].mid.x - rule[feature].high.x < dist:
                        neighbour = rule[feature]
                        dist = other_rule[feature].mid.x - rule[feature].high.x
                if neighbour is not None and dist != 0.0:
                    # merge points if necessary
                    neighbour.low = rule[feature].mid
                    rule[feature].high = neighbour.mid
        print("Repaired")

    def inspect(self):
        """
        Print rules
        """
        # find linguistic variable for each feature (list of sets of membership functions, index is feature number)
        lvs = []
        for feat_index in range(0, self._nb_of_features):
            lvs.append(set())
        for mfs in self._rules.keys():
            for feat_index in range(0, self._nb_of_features):
                lvs[feat_index].add(mfs[feat_index])

        for mfs, target_class in self._rules:



# test script
nfs = NFS(min_observations_per_rule=20)
iris = datasets.load_iris()
nfs.train(iris.data, iris.target, 100, 0.2)
nfs.inspect()