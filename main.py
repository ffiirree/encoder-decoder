import os
import time

filters = "32 64 128 256 512"
epochs = 15
lr = 0.5
patience = 8

def run_script(script:str, model:str):
    cmd = f'python {script} --model {model} --filters {filters} --epochs {epochs} --lr {lr} --patience {patience}'
    print(f'\n===\n {cmd}\n===')
    os.system(cmd)
    time.sleep(60)

if __name__ == '__main__':
    run_script('./train.py', 'ae')
    run_script('./train.py', 'unet')
    run_script('./train.py', 'unet3plus')

    run_script('./train_re.py', 'ae')
    run_script('./train_re.py', 'unet')
    run_script('./train_re.py', 'unet3plus')

    run_script('./train_re_object.py', 'ae')
    run_script('./train_re_object.py', 'unet')
    run_script('./train_re_object.py', 'unet3plus')
