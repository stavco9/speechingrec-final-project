class EditWeights:
    """
    Abstract base class for classes providing edit weights for sequence alignment operations.
    """

    def pair_weight(self, first_obj, second_obj) -> float:
        """
        Get the weight for aligning the pair of given objects.

        @param first_obj The object from the first sequence.
        @param second_obj The object from the second sequence.
        @return The match weight if the two objects are equal,
                otherwise the substitution weight.
        """
        pass

    def insertion_weight(self, obj) -> float:
        """
        Get the insertion weight of the given object.

        @param obj The inserted object in the second sequence.
        @return The insertion weight.
        """
        pass

    def deletion_weight(self, obj) -> float:
        """
        Get the deletion weight of the given object.

        @param obj The deleted object in the first sequence.
        @return The deletion weight.
        """
        pass

class GeneralEditWeights(EditWeights):
    def __init__(self, match_weight: float, substitution_weight: float, insdel_weight: float):
        self.match_weight = match_weight
        self.substitution_weight = substitution_weight
        self.insdel_weight = insdel_weight

    def pair_weight(self, first_obj, second_obj) -> float:
        if first_obj == second_obj:
            return self.match_weight
        else:
            return self.substitution_weight

    def insertion_weight(self, obj) -> float:
        return self.insdel_weight

    def deletion_weight(self, obj) -> float:
        return self.insdel_weight

class LevenshteinWeights(GeneralEditWeights):
    def __init__(self):
        super().__init__(match_weight=0, substitution_weight=-1, insdel_weight=-1)

class UniformWeights(GeneralEditWeights):
    def __init__(self):
        super().__init__(match_weight=2, substitution_weight=-1, insdel_weight=-0.75)

class NestedUniformWeights(GeneralEditWeights):
    def __init__(self):
        super().__init__(match_weight=2, substitution_weight=-1, insdel_weight=-0.75)

    def pair_weight(self, first_obj, second_obj) -> float:
        if first_obj == second_obj:
            return self.match_weight * len(first_obj)
        else:
            from modules.align_sequences import align_sequences
            return align_sequences(first_obj, second_obj, UniformWeights())[0]
    
    def insertion_weight(self, obj) -> float:
        return self.insdel_weight * len(obj)

    def deletion_weight(self, obj) -> float:
        return self.insdel_weight * len(obj)