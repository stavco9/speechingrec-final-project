from part2 import Part2
import os

def main():
    part2 = Part2(
        input_transcriptions_file=os.path.join('results', 'part1_transcriptions.tsv'),
        output_statistics_file=os.path.join('results', 'part3_statistics.csv'),
        output_transcriptions_file=os.path.join('results', 'part3_transcriptions.tsv')
    )
    statistics_total = part2.process_transcriptions(to_normalize=True)

    # Iterate over the frequent errors and print the most frequent errors
    for word_pair, num in statistics_total.frequent_errors(k=20):
        print('-> "%s" replaced by "%s" %d times.' %
        (word_pair[0], word_pair[1], num))

    part2.save_statistics(statistics_total)

if __name__ == "__main__":
    main()