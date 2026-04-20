#!/bin/bash

# step one get topos texts (takes around 1.5 hours)
# echo "########## Extracting topos texts ##########"
# python data_gathering/utils/topos_extraction.py

# step two link to shipwrecks
# echo "########## Linking to shipwrecks ##########"
# python data_gathering/utils/shipwreck_linking.py

# # step three get concordances
# echo "########## Getting concordances of texts ##########"
# python data_gathering/utils/concordance.py

# # step four filter using topic modeling
# echo "########## Filtering using topic modeling ##########"
# python data_gathering/utils/topic_model_filtering.py


# step five train+eval model
echo "########## Finetuning/Evaluating ##########"
# tensorboard --logdir /home/imruhi/Documents/RiskPerceptionSeafare/classification/mmBERT-base_finetuned/checkpoints/runs
python -m classification.finetune