`Python 3.7`
Intruction:
```Bash
sudo apt install git
git clone https://github.com/alexunderch/ReLMMinvestigation.git
cd ReLMMinvestigation/impl
chmod +x setup.sh
#just in case
export $LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
#or in colab
import os
os.environ['LD_LIBRARY_PATH'] = "/root/.mujoco/mujoco200/bin/"
./setup.sh
```

to run experiments
```Bash
xvfb-run python /content/ReLMMinvestigation/impl/ReLMM/others/main_experimental.py --render_eval #or --render_train

```
