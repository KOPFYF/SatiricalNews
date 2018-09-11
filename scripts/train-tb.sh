scratch='/homes/du113/scratch/'
basedir=$scratch'cnn-political-data/'
log=$scratch'trainerror.log'

if [ -s $log ] ; then
    echo 'deleting log file '$log
    rm $log
fi

CUDA_VISIBLE_DEVICES=2,3 python model_utils.py --embedding_file $scratch'glove100d.txt' \
    --train_path $basedir'aug_20_train_samples.csv' \
    --test_path $basedir'aug_20_test_samples.csv' \
    -e 30 \
    --cuda \
    2>$log;

if [ -s $log ] ; then
    echo ERROR
    cat $log | mail -s "An error occurred during training" du113@purdue.edu
else
   echo "training completed" | mail -s "training completed" du113@purdue.edu
fi
