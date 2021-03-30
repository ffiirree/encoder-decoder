import os
import time

os.system("python ./train.py --model='ae'")
time.sleep(60)
os.system("python ./train.py --model='unet'")
time.sleep(60)
os.system("python ./train.py --model='unet3plus'")
time.sleep(60)

os.system("python ./train_re.py --model='ae'")
time.sleep(60)
os.system("python ./train_re.py --model='unet'")
time.sleep(60)
os.system("python ./train_re.py --model='unet3plus'")
time.sleep(60)

os.system("python ./train_re_object.py --model='ae'")
time.sleep(60)
os.system("python ./train_re_object.py --model='unet'")
time.sleep(60)
os.system("python ./train_re_object.py --model='unet3plus'")
time.sleep(60)

# os.system("python ./train_re_border.py --model='ae'")
# os.system("python ./train_re_border.py --model='unet'")
# os.system("python ./train_re_border.py --model='unet3plus'")
