


1) put the firmware one that will start talking over USB
cd firmaware && pio run -t upload


2) open the collection client that starts listening on USB

cd client && uv run app.py 
(--port /dev/tty.usbmodem1101 --baud 2000000 by default - check ls /dev/tty*)

cd client && uv run app.py --test


you can now record trials that get saved under /client/recordings

3) read out and visualize the trials
cd model && uv run visualize.py



### architecture
/firmware you flash on your device and then it will stream shit over usb
/host will pick up on the reading, put it on a frontend, and let you record your shit. will create subfolder in /host/recordings for each time you click start and end session
/model 
    data takes all subfolders and turns them into a dataset. calls preprocessing functions here.
    visualize takes the samples and displays them
    data has a function to pull the events inside which correspond with labeled things. it can also just give the raw array as one, which visualize uses to pick random samples of a length when doing --raw


