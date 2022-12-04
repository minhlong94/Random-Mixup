DATA160=$1
DATA352=$2
METHOD=$3

NAME=comix

CONFIG1=configs/${NAME}/configs_fast_phase1.yml
CONFIG2=configs/${NAME}/configs_fast_phase2.yml
CONFIG3=configs/${NAME}/configs_fast_phase3.yml

PREFIX1=fast_phase1_${NAME}
PREFIX2=fast_phase2_${NAME}
PREFIX3=fast_phase3_${NAME}

OUT1=fast_train_phase1_${NAME}.out
OUT2=fast_train_phase2_${NAME}.out
OUT3=fast_train_phase3_${NAME}.out

EVAL1=fast_eval_phase1_${NAME}.out
EVAL2=fast_eval_phase2_${NAME}.out
EVAL3=fast_eval_phase3_${NAME}.out


END1=./trained_models/fast_phase1_${NAME}/checkpoint.pth.tar
END2=./trained_models/fast_phase2_${NAME}/checkpoint.pth.tar
END3=./trained_models/fast_phase3_${NAME}/checkpoint.pth.tar

# training for phase 1
python3 -u main_fast_ddp.py $DATA160 -c $CONFIG1 --method $METHOD --output_prefix $PREFIX1 | tee $OUT1

# evaluation for phase 1
#python -u main_fast_ddp.py $DATA352 -c $CONFIG1 --method $METHOD --output_prefix $PREFIX1 --resume $END1  --evaluate | tee $EVAL1

# training for phase 2
python3 -u main_fast_ddp.py $DATA352 -c $CONFIG2 --method $METHOD --output_prefix $PREFIX2 --resume $END1 | tee $OUT2

# evaluation for phase 2
#python -u main_fast_ddp.py $DATA352 -c $CONFIG2 --num_patches $NUMPATCH --output_prefix $PREFIX2 --resume $END2 --evaluate | tee $EVAL2

# training for phase 3
python3 -u main_fast_ddp.py $DATA352 -c $CONFIG3 --method $METHOD --output_prefix $PREFIX3 --resume $END2 | tee $OUT3

# evaluation for phase 3
python3 -u main_fast_ddp.py $DATA352 -c $CONFIG3  --output_prefix $PREFIX3 --resume $END3 --evaluate | tee $EVAL3

