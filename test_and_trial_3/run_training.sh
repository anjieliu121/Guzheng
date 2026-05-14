#!/bin/bash
cd /Users/anjie/Documents/MyGuzheng/Guzheng/test_and_trial_2
python3 -u 04_train.py --epochs 200 > logs/training_output.log 2>&1
echo "EXIT CODE: $?" >> logs/training_output.log
