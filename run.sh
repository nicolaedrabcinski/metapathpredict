#!/bin/bash
python /home/nicolaedrabcinski/metapathpredict/workflow/prepare_ds.py /home/nicolaedrabcinski/metapathpredict/main_config.json
echo "Finished preparation of training datasets"

python /home/nicolaedrabcinski/metapathpredict/workflow/train.py /home/nicolaedrabcinski/metapathpredict/main_config.json
echo "Finished training"

python /home/nicolaedrabcinski/metapathpredict/workflow/predict.py /home/nicolaedrabcinski/metapathpredict/main_config.json
echo "Finished prediction"

echo "If there were no errors, project works properly!"
