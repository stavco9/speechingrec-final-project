from IPython.display import display 
import pandas as pd
import os

from align_sequences import align_sequences, get_difference
from edit_weights import LevenshteinWeights, UniformWeights, NestedUniformWeights

print("="*50)
print("Part1: Two string alignment")
print("="*50)
str1 = "compassion"
str2 = "comparison"
print("Comparing alignments of by levenshtein weights: ", str1, "and", str2)
align_score, aligned_pairs = align_sequences(str1, str2, LevenshteinWeights())
print("Alignment score: ", align_score)
print("Aligned pairs: ", aligned_pairs)
print("Differences: ", get_difference(aligned_pairs))
print("Count of differences: ", len(get_difference(aligned_pairs)))

print("="*50)
print("Part2: Two sequence alignment")
print("="*50)
seq1 = "friends romans countrymen lend me your ears i come to bury caesar not to praise him".split()
seq2 = "my friends romans country men land me your ears i come to bury caesar note raise".split()
print("Comparing alignments of by levenshtein weights: ", seq1, "and", seq2)
align_score, aligned_pairs = align_sequences(seq1, seq2, LevenshteinWeights())
print("Alignment score: ", align_score)
print("Aligned pairs: ", aligned_pairs)
print("Differences: ", get_difference(aligned_pairs))
print("Count of differences: ", len(get_difference(aligned_pairs)))

print("="*50)
print("Part3: Uniform weights")
print("="*50)
str1 = "compassion"
str2 = "comparison"
print("Comparing alignments of by uniform weights: ", str1, "and", str2)
align_score, aligned_pairs = align_sequences(str1, str2, UniformWeights())
print("Alignment score: ", align_score)
print("Aligned pairs: ", aligned_pairs)
print("Differences: ", get_difference(aligned_pairs))
print("Count of differences: ", len(get_difference(aligned_pairs)))

print("="*50)
print("Part4: Nested uniform weights")
print("="*50)
seq1 = "friends romans countrymen lend me your ears i come to bury caesar not to praise him".split()
seq2 = "my friends romans country men land me your ears i come to bury caesar note raise".split()
print("Comparing alignments of by nested uniform weights: ", seq1, "and", seq2)
align_score, aligned_pairs = align_sequences(seq1, seq2, NestedUniformWeights())
print("Alignment score: ", align_score)
print("Aligned pairs: ", aligned_pairs)
print("Differences: ", get_difference(aligned_pairs))
print("Count of differences: ", len(get_difference(aligned_pairs)))

#df = pd.read_csv(os.path.join('..', 'transcriptions.tsv'), sep='\t')

#display(df)