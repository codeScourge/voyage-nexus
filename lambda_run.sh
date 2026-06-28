ssh ubuntu@170.9.24.65 "mkdir -p ~/voyage-nexus/model"
scp model/tmspd_loader.py model/tmspd_smoke.py model/loso_align.py model/loso.py model/data.py model/train.py model/_preprocessors.py model/_viewer_core.py model/row2_feature_fusion.py model/row3_late_fusion.py ubuntu@170.9.24.65:~/voyage-nexus/model/
ssh ubuntu@170.9.24.65
pip install torch numpy scipy scikit-learn tqdm mne curryreader --break-system-packages -q
DATA="~/voyage-nexus/T-MSPD/V1/multidimensional_physiological_signals/2.raw_data"
cd ~/voyage-nexus/model
tmux new -s tmspd

ssh ubuntu@170.9.24.65 "mkdir -p ~/Voyage/model ~/Voyage/T-MSPD/V1/multidimensional_physiological_signals"

great@PDYoga MINGW64 /e/Work/Voyage_Interfaces/voyage-nexus/T-MSPD/V1/multidimensional physiological signals (main)
$ scp -r 2.raw_data ubuntu@170.9.24.65:~/Voyage/T-MSPD/V1/multidimensional_physiological_signals

export REPO=~/voyage-nexus/model      # where loso.py loso_align.py train.py data.py live
export DATA=~/Voyage/T-MSPD/V1/multidimensional_physiological_signals/2.raw_data
echo "REPO=$REPO"; echo "DATA=$DATA"
ssh ubuntu@170.9.24.65 "mkdir -p ~/voyage-nexus/model ~/Voyage/T-MSPD/V1/multidimensional_physiological_signals"
scp model/tmspd_loader.py model/tmspd_smoke.py model/loso_align.py model/loso.py model/data.py model/train.py model/_preprocessors.py model/_viewer_core.py model/row2_feature_fusion.py model/row3_late_fusion.py ubuntu@170.9.24.65:~/voyage-nexus/model/
cd "/e/Work/Voyage_Interfaces/voyage-nexus/T-MSPD/V1/multidimensional physiological signals"
for s in $(seq -w 1 5); do
  scp -r "2.raw_data/overt speech/S0$s" \
    ubuntu@<box>:"$DATA/overt speech/"          # start with OVERT (best SNR to debug on)
done
ssh ubuntu

ubuntu@64-181-255-118:~/voyage-nexus/model$ stdbuf -oL python -u tmspd_smoke.py --root ~/Voyage/T-MSPD/V1/multidimensional_physiological_signals/2.raw_data --mode "silent speech"
    --modality fusion --subjects 1-30 --epochs 60 --shots 5 \
    2>&1 | tee tmspd_silent_fusion.log

# (2) montage-HONEST EEG arm (peri-auricular). Shares the cache from run 1 -> starts instantly.
stdbuf -oL python -u tmspd_smoke.py --root ~/Voyage/T-MSPD/V1/multidimensional_physiological_signals/2.raw_data --mode "overt speech" \
    --modality eeg --subjects 1-15 --epochs 60 --shots 5 \
    2>&1 | tee tmspd_silent_eeg.log

# (3) EMG arm (algorithm prior). Also shares the cache.
stdbuf -oL python -u tmspd_smoke.py --root ~/Voyage/T-MSPD/V1/multidimensional_physiological_signals/2.raw_data --mode "overt speech" \
    --modality emg --subjects 1-15 --epochs 60 --shots 5 \
    2>&1 | tee tmspd_silent_emg.log

export REPO=~/voyage-nexus/model      # where loso.py loso_align.py train.py data.py live
export DATA=~/Voyage/T-MSPD/V1/multidimensional_physiological_signals/2.raw_data
echo "REPO=$REPO"; echo "DATA=$DATA"

stdbuf -oL python -u tmspd_smoke.py --root "$DATA" --mode "silent speech"
  --modality eeg --subjects 2-30 --epochs 60 --shots 5     2>&1 | tee tmspd_silent_eeg.log

## Pretrain Run