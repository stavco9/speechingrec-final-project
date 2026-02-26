from modules.align_sequences import align_sequences
from modules.edit_weights import NestedUniformWeights
from collections import Counter
from typing import List, Tuple

class AccuracyStatistics:
    def __init__(self, reference_text: List[str]=[], transcribed_text: List[str]=[]):
        self.aligned_score, self.aligned_pairs = align_sequences(reference_text, transcribed_text, NestedUniformWeights())
        self.all_differences = self.get_difference()
        self.N_gt = len(reference_text)
        self.N_asr = len(transcribed_text)
        self.M = len(self.get_matches())
        self.S = len(self.get_substitutions())
        self.I = len(self.get_insertions())
        self.D = len(self.get_deletions())
        self.wer = self.get_wer()
        self.recall = self.get_recall()
        self.precision = self.get_precision()
        self.f1_score = self.get_f1_score()

    def frequent_errors(self, k: int = 0) -> List[Tuple]:
        if k == 0:
            k = len(self.all_differences)
        errors = dict[Tuple[str, str], int](Counter[Tuple[str, str]](self.all_differences))
        errors = dict[Tuple[str, str], int](sorted(errors.items(), key=lambda x: x[1], reverse=True))
        word_pairs = [(error[0], error[1]) for error in errors.keys()]
        word_pairs_cnt = [error for error in errors.values()]
        return list[Tuple[str, str], int](zip(word_pairs, word_pairs_cnt))[:k]

    def get_difference(self) -> List[Tuple]:
        differences = [(a if a is not None else "", b if b is not None else "") 
            for a, b in self.aligned_pairs if a != b]
        return differences

    def get_matches(self) -> List[Tuple]:
        return [(a, b) for a, b in self.aligned_pairs if a == b]

    def get_substitutions(self) -> List[Tuple]:
        return [(a, b) for a, b in self.aligned_pairs if a != b and a is not None and b is not None]

    def get_insertions(self) -> List[Tuple]:
        return [("", b) for a, b in self.aligned_pairs if a is None and b is not None]

    def get_deletions(self) -> List[Tuple]:
        return [(a, "") for a, b in self.aligned_pairs if a is not None and b is None]

    def get_wer(self) -> float:
        return (self.S + self.I + self.D) / self.N_gt if self.N_gt > 0 else 0

    def get_recall(self) -> float:
        return self.M / self.N_gt if self.N_gt > 0 else 0
    
    def get_precision(self) -> float:
        return self.M / self.N_asr if self.N_asr > 0 else 0
    
    def get_f1_score(self) -> float:
        return (2 * self.M) / (self.N_gt + self.N_asr) if self.N_gt + self.N_asr > 0 else 0

    def __iadd__(self, other: 'AccuracyStatistics') -> 'AccuracyStatistics':
        self.N_gt += other.N_gt
        self.N_asr += other.N_asr
        self.M += other.M
        self.S += other.S
        self.I += other.I
        self.D += other.D
        self.wer += other.wer
        self.recall += other.recall
        self.precision += other.precision
        self.f1_score += other.f1_score
        self.all_differences.extend(other.all_differences)
        return self

    def to_dict(self) -> dict:
        return {
            'N_gt': self.N_gt,
            'N_asr': self.N_asr,
            'M': self.M,
            'S': self.S,
            'I': self.I,
            'D': self.D,
            'wer': self.wer,
            'recall': self.recall,
            'precision': self.precision,
            'f1_score': self.f1_score
        }