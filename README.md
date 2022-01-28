Environment setup:

1. To use MuJoCo environments you need to download this archive:
https://roboti.us/download/mjpro150_linux.zip
Then unpack it into the folder 
$HOME/.mujoco/mjpro150
Also you need to get a license key from 
https://roboti.us/file/mjkey.txt
and put it into
$HOME/.mujoco/mjkey.txt
2. conda env create -f environment.yml
3. conda activate adatqc

After that you can run:
bash run_adatqc.sh

This script will consecutively run one instance of training for each environment.

Results will be saved in './data'. Metrics data is stored in files 'metric.json'. One line of such file is stored in json dictionary with metrics values, they are generated once in every 1000 steps of training.
