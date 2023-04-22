# Group AdaBoost with Fairness Constraint

The source codes for paper 'Group AdaBoost with Fairness Constraint' accepted by SDM 23

[Paper]()

## Architecture
<img width="1087" alt="FIG" src="https://user-images.githubusercontent.com/41327917/209700630-39c23668-f1f8-40cb-b244-89691c197ab9.png">

## Setup

```setup
pip install fairlearn 
pip install torch

cd real_world_data/DataPreprocessing
unzip kdd.zip
```

## Running
### synthetic data
```
cd synthetic_data
vis_boosting.ipynb
```
### real-world data
```
cd real_world_data
# For every sh file, change <your-file-path> to the log path you prefer

# Run Baselines via Vallina Evaluating Benchmarks
sh run_baseline.sh
# Run Baselines via Converted Evaluating Benchmarks 
sh run_baseline_dif_imbalance.sh

# Run Our Method and Adaboost via Vallina Evaluating Benchmarks 
sh run_group_adaboost.sh
# Run Our Method and Adaboost via Converted Evaluating Benchmarks
sh run_group_adaboost_dif_imbalance.sh
```


## Reference
[AdaFair](https://github.com/iosifidisvasileios/AdaFair.git) 

