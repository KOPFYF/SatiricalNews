scratch='/homes/du113/scratch/'
basedir=$scratch'cnn-political-data/'
log=$scratch'testerror.log'

if [ -s $log ] ; then
    echo 'deleting log file '$log
    rm $log
fi

CUDA_VISIBLE_DEVICES=1,2,4 python test_baseline.py --embedding_file $basedir'pretrained_embed.pth.tar' \
    --num_workers=2 \
    --test_path $basedir'aug_20_test_samples.csv' \
    --cuda \
    2>$log;

if [ -s $log ] ; then
    echo "ERROR: error occurred"
    cat $log | mail -s "An error occurred during testing" du113@purdue.edu
else
   echo "testing completed" | mail -s "testing completed" du113@purdue.edu
fi

