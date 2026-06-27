


TODO
- debug until next iteration, downgrade torch, retrain here first
- 1830 start modality research
- scp or not, deploy on runpod for next big run 


### using
1) put the firmware one that will start talking over USB
cd firmaware && pio run -t upload


2) open the collection client that starts listening on USB

cd client && uv run app.py 
(--port /dev/tty.usbmodem1101 --baud 2000000 by default - check ls /dev/tty*)

cd client && uv run app.py --test


you can now record trials that get saved under /client/recordings

3) read out and visualize the trials
cd model && uv run visualize.py


## explanation
we have built an EEG/EMG device and now are training a silent-speech model on it.

### data collection
/firmware you flash on your device and then it will stream shit over usb

/client will pick up on the reading, put it on a frontend, and let you record your shit. will create subfolder in /client/recordings for each session you record at

---

you have three modes
- scramble-fast: absolute fastest bits per seconds on just speaking words - smaller break between same words than different
- scramble-breaks: the wait between word is alway 1.6s (same length as a word window) between, so we can use that for silence

### training
in data.py you can decide whether to include silence and unknown-word classes (and which trial modes contribute silence), as well as the type of preprocessing
in train.py you choose channels to use 

here also the cutting is handled: to construct word_starting, and word_ending, we sample somewhere silence and word, and then based on where it lands
- >75% word = word label
- 25-75% word = starting / ending
- <25% word = silent

add --continue flag, will continue training with the currently set rules, and create a folder with the same name but `-continued` attached to it
TODO: for loss decresion

### dataset
Silence: `INCLUDE_SILENCE_LABEL` (master), plus `INCLUDE_SCRAMBLE_BREAKS_SILENCE` and `INCLUDE_NEGATIVE_LABELS_SILENCE` for per-trial sources.

theres the simple fast approach, which just first cuts away x/y sessions for val, and another one that tries different combinations so that the distribution follows in the end what we want it to have

we have a dict that limits each label to x% (after put into sets) - algorithm tries to condense diversity, by dropping things while leaving at least some from a session block and session

#### ivan, collected on gabriella
- 2026-06-25_00-03-53_session_5f75476f, 2026-06-25_00-10-59_session_c019258d - collected slightly sweaty after run, was doing little head nodding half the time
- 2026-06-25_06-50-40_session_dda4a668 - was laughing at last few samples, might need to delete the latest scrable
- 2026-06-26_00-09-06_session_9faeac69, 2026-06-26_00-10-59_session_0dbb211f, 