# Speech Recognition Final Project

## Submittors

- **Stav Cohen** - 316492776
- **Ron Kozitsa** - 312544240

## Project Structure

### Source Code

- `part1.py` - Part 1 implementation
- `part2.py` - Part 2 implementation
- `part3.py` - Part 3 implementation
- `part4.py` - Part 4 implementation
- `consts/` - Constants and configuration files
  - `correction_dict.py` - Dictionary for text corrections
- `modules/` - Core modules
  - `accuracy_statistics.py` - Accuracy statistics calculations
  - `align_sequences.py` - Sequence alignment functionality
  - `edit_weights.py` - Edit weights for alignment
  - `normalize_text.py` - Text normalization module
  - `statistics_df.py` - Statistics DataFrame utilities

### Outputs (CSV / TSV)

All output files are located in the `results/` folder:

- `part1_transcriptions.tsv` - Part 1 transcription results
- `part2_statistics.csv` - Part 2 accuracy statistics
- `part3_normalized_transcriptions.tsv` - Part 3 normalized transcriptions
- `part3_statistics.csv` - Part 3 statistics
- `part4_augmentation_log.tsv` - Part 4 augmentation log
- `part4_noisy_transcriptions.tsv` - Part 4 noisy transcriptions
- `part4_normalized_transcriptions.tsv` - Part 4 normalized transcriptions
- `part4_statistics.csv` - Part 4 statistics

### Part 3 Report

The project report is located in the `reports/` folder:

- `Speaking recognition final project report.pdf`

## Requirements

See `requirements.txt` for Python dependencies.