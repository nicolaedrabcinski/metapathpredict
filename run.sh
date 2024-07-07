#!/bin/bash
python /home/nicolaedrabcinski/eebg_project/workflow/prepare_ds.py /home/nicolaedrabcinski/eebg_project/main_config.json
echo "finished preparation of training dataset for neural network and random forest modules"

python /home/nicolaedrabcinski/eebg_project/workflow/train.py /home/nicolaedrabcinski/eebg_project/main_config.json
echo "finished training neural network and random forest modules"

python /home/nicolaedrabcinski/eebg_project/workflow/predict.py /home/nicolaedrabcinski/eebg_project/main_config.json
echo "finished prediction of the test file"

echo "If there were no errors, eebg_project works properly!"