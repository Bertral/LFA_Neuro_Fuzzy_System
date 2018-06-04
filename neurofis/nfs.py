import numpy as np
import itertools
import matplotlib.pyplot as plt
import sklearn.metrics
from neurofis.point import Point
from neurofis.mf import MF
from fuzzy_systems.core.rules.fuzzy_rule import FuzzyRule
from fuzzy_systems.core.fis.singleton_fis import SingletonFIS
from fuzzy_systems.core.membership_functions.singleton_mf import SingletonMF
from fuzzy_systems.core.membership_functions.trap_mf import TrapMF
from fuzzy_systems.core.linguistic_variables.linguistic_variable import LinguisticVariable
from fuzzy_systems.core.membership_functions.lin_piece_wise_mf import LinPWMF
from fuzzy_systems.core.rules.fuzzy_rule import FuzzyRule, Antecedent, Consequent
from fuzzy_systems.core.fis.fis import FIS, OR_max, AND_min, MIN, COA_func
from fuzzy_systems.view.fis_viewer import FISViewer
from fuzzy_systems.view.fis_surface import show_surface
from sklearn import datasets
from sklearn.utils import shuffle


class NFS:
    def __init__(self, max_rules: int = 7, min_observations_per_rule: int = 5):
        self._nb_of_features = 0
        self._max_rules = max_rules
        self._min_observations_per_rule = min_observations_per_rule
        self._rules = None

    def train(self, data: np.ndarray, target: np.ndarray, nb_iter: int, learning_rate: float):
        """
        Train the NFS for nb_iter complete passes on data
        Data must be a 2-dimentionnal numpy matrix (each row is an observation, each col a feature)

        """
        self._rules = {
        }  # dictionnary (key = tuple of fuzzy sets, value = dominant class)
        self._nb_of_features = np.shape(data)[1]

        # shuffling reduces the risks of having rules override each other while training
        data, target = shuffle(data, target)

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
        # list of tuples of fuzzy sets (every case of the grid)
        intersections = list(itertools.product(*mfs))
        print("Rules without consequent built : " + str(len(intersections)))

        print("Finding rule consequents and removing weak rules ...")
        # for each square, add a rule for the highest class if it has enough observations
        for intersection in intersections:
            classes = {}
            for observ in range(0, np.shape(data)[0]):
                found = True
                # every feature must be found
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
            nb_of_observations, rule_class = max(
                zip(classes.values(), classes.keys()))
            if nb_of_observations >= self._min_observations_per_rule:
                # use this rule
                self._rules[intersection] = rule_class
        print("Rules found : " + str(len(self._rules)))

        self.repair(np.shape(data)[1])

        print("Training ...")
        for _ in range(0, nb_iter):
            for obs in range(0, np.shape(data)[0]):
                # find the most activated rule for this observation
                max_rule = None
                max_act = 0
                for mfs, target_class in self._rules.items():
                    act = 1
                    # activate
                    for feat in range(0, len(mfs)):
                        act = min(act, mfs[feat].fuzzyfy(data[obs, feat]))
                    # compare activation with max_act
                    if act >= max_act:
                        max_rule = (mfs, target_class)
                        max_act = act
                if max_rule is None:
                    continue  # skip if the observation has no rule
                # adjust membership functions for the most activated rule baseed on this observation
                for feat in range(0, len(max_rule[0])):
                    # move membership function to/away from (if same/different class) data[obs, feat] on distance
                    # learning_rate
                    max_rule[0][feat].move(
                        data[obs, feat], learning_rate, max_rule[1] == target[obs])

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
                    if rule[feature] is not None and rule[feature].high.x < other_rule[feature].mid.x \
                            and other_rule[feature].mid.x - rule[feature].high.x < dist:
                        neighbour = other_rule[feature]
                        dist = other_rule[feature].mid.x - rule[feature].high.x
                if neighbour is not None and dist != 0.0:
                    # merge points if necessary
                    neighbour.low = rule[feature].mid
                    rule[feature].high = neighbour.mid
        print("Repaired")

    def pruning(self, data: np.ndarray):
        "Remove antecedents that are not used in rules and poorly used rules"

        print("Number of rules before pruning :", len(self._rules))

        # track usage of rules for pruning
        _rules_usage = {}
        # track antecedent usage for each rules
        _antecedent_usage = {}
        for obs in range(0, np.shape(data)[0]):
            # find the most activated rule for this observation
            max_act = 0
            max_mfs = None
            for mfs in self._rules.keys():
                # add the rule to _rules_usage dict if not already in
                if mfs not in _rules_usage:
                    _rules_usage[mfs] = 0

                # add the rule's antecedents to _antecedent_usage if not already in
                if mfs not in _antecedent_usage:
                    _antecedent_usage[mfs] = {}

                act = 1
                min_ant = None
                for ant in range(0, len(mfs)):
                    # add the antecedent to this rule's _antecedent_usage if not already in
                    if mfs[ant] not in _antecedent_usage[mfs]:
                        _antecedent_usage[mfs][mfs[ant]] = 0

                    mf_act = mfs[ant].fuzzyfy(data[obs, ant])
                    if mf_act <= act:
                        act = mf_act
                        min_ant = mfs[ant]

                # save the fact that this antecedent is dominant
                # for this rule for this observation
                _antecedent_usage[mfs][min_ant] += 1

                if act >= max_act:
                    max_act = act
                    max_mfs = mfs
            # save the fact that this rule is used for this observation
            _rules_usage[max_mfs] += 1

        # sort rule by descending usage order
        best_rules_usage = [(k, _rules_usage[k]) for k in
                            sorted(_rules_usage, key=_rules_usage.get, reverse=True)][:self._max_rules]

        best_rules = {}
        for mfs, _ in best_rules_usage:
            used_mfs = []
            for ant in mfs:
                # build rule with only antecedents that have been used at least once
                if _antecedent_usage[mfs][ant] > 0:
                    used_mfs.append(ant)
                else:
                    used_mfs.append(None)
                    print("antecedent removed")
            assert len(used_mfs) > 0
            best_rules[tuple(used_mfs)] = self._rules[mfs]

        self._rules = best_rules

        print("Number of rules after pruning :", len(self._rules))

        # start by checking for holes
        self.repair(np.shape(data)[1])

    def test(self, data: np.ndarray, target: np.ndarray):
        """
        Test the trained model
        :param data:
        :param target:
        :return:
        """

        predictions = []
        for obs in range(0, np.shape(data)[0]):
            # find the most activated rule for this observation
            max_act = 0
            max_class = -1
            for mfs, target_class in self._rules.items():
                act = 1
                # activate
                for feat in range(0, len(mfs)):
                    act = min(act, mfs[feat].fuzzyfy(data[obs, feat]))
                if act >= max_act:
                    max_class = target_class
                    max_act = act
            predictions.append(max_class)

        print("Confusion matrix : " +
              str(sklearn.metrics.confusion_matrix(target, predictions)))
        print("Accuracy score : " +
              str(sklearn.metrics.accuracy_score(target, predictions)))
        print("Precision : " + str(sklearn.metrics.precision_score(target,
                                                                   predictions, average='micro')))
        print("Recall : " + str(sklearn.metrics.recall_score(target,
                                                             predictions, average='micro')))

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

        displayable_rules = []
        '''
        for mfs, target_class in self._rules:
            ling_values_dict = {}
            for feat_index in range(0, self._nb_of_features):
                ling_values_dict[""]
            displayable_lvs.append(LinguisticVariable(name="feature"+str(feat_index), ling_values_dict={
                "": TrapMF(mfs[feat_index].low.x, mfs[feat_index].mid.x, mfs[feat_index].mid.x, mfs[feat_index].high.x),
            }))
        fis = FIS(
            rules=displayable_rules,
            aggr_func=np.max,
            defuzz_func=COA_func
        )

        fisv = FISViewer(fis, figsize=(12, 10))
        fisv.show()
        '''
