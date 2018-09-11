scratch='/homes/du113/scratch/'
basedir=$scratch'cnn-political-data/'
log=$scratch'trainerror.log'

if [ -s $log ] ; then
    echo 'deleting log file '$log
    rm $log
fi

CUDA_VISIBLE_DEVICES=1,2,4 python Model_baseline_ugly.py --embedding $scratch'glove100d.txt' \
    --train_path $basedir'aug_20_train_samples.csv' \
    -e 10 \
    --cuda \
    2>$log;

if [ -s $log ] ; then
    cat $log | mail -s "An error occurred during training" du113@purdue.edu
else
   echo "training completed" | mail -s "training completed" du113@purdue.edu
fi
