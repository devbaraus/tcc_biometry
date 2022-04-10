#!/bin/bash

overlap=(0 0.5)
coeff=(40 34 26 18)
segment=(1 2 3)
bases=(base_portuguese)
# bases=(base_portuguese_20)
# bases=(base_portuguese_4 base_portuguese_20)

# python3 /src/tcc/main.py -c 40 -s 1 -o 0.5 -a 30 -b base_portuguese

# for i in ${!overlap[@]}; do
#     for j in ${!coeff[@]}; do
#         for k in ${!segment[@]}; do
#             for l in ${!bases[@]}; do
#                 # echo "python3 /src/tcc/main.py -c ${coeff[$j]} -s ${segment[$k]} -o ${overlap[$i]} -a 30 -b ${bases[$l]}"
#                 python3 /src/tcc/main.py -c ${coeff[$j]} -s ${segment[$k]} -o ${overlap[$i]} -a 30 -b ${bases[$l]}
#             done
#         done
#     done
# done


# python3 /src/tcc/inference.py -m /src/tcc/models/base_portuguese_20/SEG_1_OVERLAP_0_AUG_30/MFCC_18/D60_DO0_D0_DO0_D0/1649338837_86.86131238937378

for w in ${!bases[@]}; do
    PARENT_FOLDER="/src/tcc/models/${bases[$w]}"

    find $PARENT_FOLDER -name inference -exec rm -rf {} \;

    OUTPUT=$(python3 /src/tcc/backlog/run_folder.py -f $PARENT_FOLDER)

    for i in $OUTPUT;
    do
        python3 /src/tcc/inference.py -m "$PARENT_FOLDER/$i" -i "/src/tcc/dataset/inference/${bases[$w]}"
    done
done

# find /src/tcc/models/base_portuguese_20 -name inference -exec rm -rf {} \;

# PARENT_FOLDER='/src/tcc/models/base_portuguese_20/'

# OUTPUT=$(python3 /src/tcc/backlog/run_folder.py -f $PARENT_FOLDER)

# for i in $OUTPUT;
# do
#     python3 /src/tcc/inference.py -m "$PARENT_FOLDER$i" -i /src/tcc/dataset/inference/base_portuguese_20
# done