import numpy as np
from src.fuzzy_systems.core.rules.fuzzy_rule import FuzzyRule

class NFS():
    def __init__(self, max_rules: int, min_observations_per_rule: int):
        self._max_rules = max_rules
        self._min_observations_per_rule = min_observations_per_rule

        
    def train(self, data, nb_iter):
        """
        Train the NFS for nb_iter complete passes on data
        Data must be a 2-dimentionnal numpy array of this format :
        
        value_A_of_1    value_B_of_1    ...     value_X_of_1    class_of_1
        value_A_of_2    value_B_of_2    ...     value_X_of_2    class_of_2
        ...             ...             ...     ...             ...
        value_A_of_3    value_B_of_3    ...     value_X_of_3    class_of_3
        
        """
        
        
    
    def inspect(self):
        "nothing yet"
        # TODO : print rules
