# DATA_FILE="/homes/du113/scratch/cnn-300-data/CNN_2018_8_21.txt"
# DATA_FILE="/homes/du113/scratch/cnn-political-data/CNN_2018_8_16.txt"
# DATA_FILE="/homes/du113/scratch/real-bbc/0_BBCNews.txt"
# DATA_FILE="/homes/du113/scratch/11k-data/uk/true/test.txt"
LOG="/homes/du113/scratch/echain-log.txt"
WORD_DICT="/homes/du113/scratch/170k/word_dict.pkl"
SAVE_DIR="/homes/du113/scratch/170k/"
# SAVE_DIR="/homes/du113/scratch/cnn-300-data/"
# DNAME="bbc0_1_"
# VERT_FILE="/homes/du113/scratch/test-11k/"$DNAME"verts.pkl"
# COREF_FILE="/homes/du113/scratch/test-11k/"$DNAME"corefs.pkl"
# CHAIN_FILE="/homes/du113/scratch/test-11k/"$DNAME"chains.pkl"

tdir="/homes/du113/scratch/data/text/true"
fdir="/homes/du113/scratch/data/text/fake"

if [ -s $LOG ] ; then
    echo "deleting previous error log"
    rm $LOG
fi

echo "processing "$DATA_FILE

python -W ignore event_chain.py \
    --word_dict $WORD_DICT \
    --tdir $tdir \
    --fdir $fdir \
    --save_dir $SAVE_DIR \
    2>$LOG;

if [ -s $LOG ] ; then
    echo "error occurred, sending email to user ..."
    cat $LOG | mail -s "An error occurred while building event chain" du113@purdue.edu
else
    echo "CNN completed" | mail -s "task completed without error" du113@purdue.edu
fi
