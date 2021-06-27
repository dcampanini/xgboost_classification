# ambiente de train con AI platform y test local

$virtualenv --python=/usr/bin/python2.7 env_teco1
$source env_teco1/bin/activate
$pip install -r req.txt
$pip install xgboost

# en ai platform para las veces posteriores a la creacion del ambiente
$ cd teco
$ source env_teco1/bin/activate

# en cloud shell (no tengo ambiente)
$cd census_training
$bash run_train.sh