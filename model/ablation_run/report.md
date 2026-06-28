# Electrode ablation report

Legend: positions map to physical electrodes as:
  EEG 0->EEG2, 1->EEG4, 2->EEG5, 3->EEG7, 4->EEG8, 5->EEG9, 6->EEG10, 7->EEG12
  EMG 0->EMG1, 1->EMG3, 2->EMG4, 3->EMG6, 4->EMG8, 5->EMG10, 6->EMG14, 7->EMG16

## Greedy backward-elimination path
| step | montage | balacc | std | kappa | macroF1 | chance |
|---|---|---|---|---|---|---|
| full | E8+M8 | 0.833 | 0.166 | 0.807 | 0.870 | 0.333 |
| drop EMG8 | E8+M7 | 0.851 | 0.153 | 0.830 | 0.886 | 0.333 |
| drop EEG8 | E7+M7 | 0.863 | 0.143 | 0.843 | 0.895 | 0.333 |
| drop EEG12 | E6+M7 | 0.863 | 0.122 | 0.826 | 0.883 | 0.333 |
| drop EMG6 | E6+M6 | 0.855 | 0.148 | 0.821 | 0.880 | 0.333 |
| drop EEG10 | E5+M6 | 0.866 | 0.155 | 0.855 | 0.903 | 0.333 |
| drop EMG10 | E5+M5 | 0.869 | 0.141 | 0.840 | 0.893 | 0.333 |
| drop EEG5 | E4+M5 | 0.866 | 0.120 | 0.837 | 0.891 | 0.333 |
| drop EMG1 | E4+M4 | 0.867 | 0.130 | 0.842 | 0.894 | 0.333 |
| drop EEG2 | E3+M4 | 0.863 | 0.151 | 0.843 | 0.895 | 0.333 |
| drop EMG3 | E3+M3 | 0.858 | 0.132 | 0.828 | 0.884 | 0.333 |

**Greedy final montage:** EEG ['EEG4', 'EEG7', 'EEG9']  EMG ['EMG4', 'EMG14', 'EMG16']

## Accuracy vs total electrodes (pick the knee)
| total ch | E+M | balacc | std |
|---|---|---|---|
| 16 | 8+8 | 0.833 | 0.166 |
| 15 | 8+7 | 0.851 | 0.153 |
| 14 | 7+7 | 0.863 | 0.143 |
| 13 | 6+7 | 0.863 | 0.122 |
| 12 | 6+6 | 0.855 | 0.148 |
| 11 | 5+6 | 0.866 | 0.155 |
| 10 | 5+5 | 0.869 | 0.141 |
| 9 | 4+5 | 0.866 | 0.120 |
| 8 | 4+4 | 0.867 | 0.130 |
| 7 | 3+4 | 0.863 | 0.151 |
| 6 | 3+3 | 0.858 | 0.132 |
