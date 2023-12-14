import subprocess
import time

for i in range(2020,2021):
    for j in range(4,20):
        cmd = 'python .\data.py -y '+str(i)+' -t '+str(j)
        output = subprocess.getoutput(cmd)
        print(output)