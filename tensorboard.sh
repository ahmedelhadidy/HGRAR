run_id="$1"
modelname="$2"
#param=""
#for dl in $(find ~/hygrar/$run_id -type d -iregex '.*_tensorboard' | grep perceptron_ar1.csv_1)
#do
#  modelname=$(basename $dl)
#  param+="$modelname:$dl,"
#done
#param=$(echo $param | sed 's/.$//')
#echo $param
tensorboard --logdir $(find ~/hygrar/$run_id -type d -iregex ".*${modelname}.*_tensorboard")
