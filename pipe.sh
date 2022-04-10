####
for i in {0 0.5}
# # 1 SEGUNDO
# python3 /src/tcc/main.py -c 40 -s 1 -o 0.5 -a 30 -b base_portuguese
# python3 /src/tcc/main.py -c 34 -s 1 -o 0.5 -a 30 -b base_portuguese
# python3 /src/tcc/main.py -c 26 -s 1 -o 0.5 -a 30 -b base_portuguese
# python3 /src/tcc/main.py -c 18 -s 1 -o 0.5 -a 30 -b base_portuguese

# # 2 SEGUNDOS
# python3 /src/tcc/main.py -c 40 -s 2 -o 0.5 -a 30 -b base_portuguese
# python3 /src/tcc/main.py -c 34 -s 2 -o 0.5 -a 30 -b base_portuguese
# python3 /src/tcc/main.py -c 26 -s 2 -o 0.5 -a 30 -b base_portuguese
# python3 /src/tcc/main.py -c 18 -s 2 -o 0.5 -a 30 -b base_portuguese

# # 3 SEGUNDOS
# python3 /src/tcc/main.py -c 40 -s 3 -o 0.5 -a 30 -b base_portuguese
# python3 /src/tcc/main.py -c 34 -s 3 -o 0.5 -a 30 -b base_portuguese
# python3 /src/tcc/main.py -c 26 -s 3 -o 0.5 -a 30 -b base_portuguese
# python3 /src/tcc/main.py -c 18 -s 3 -o 0.5 -a 30 -b base_portuguese


# ####
# # 1 SEGUNDO
# python3 /src/tcc/main.py -c 40 -s 1 -o 0.5 -a 30 -b base_portuguese_40
# python3 /src/tcc/main.py -c 34 -s 1 -o 0.5 -a 30 -b base_portuguese_40
# python3 /src/tcc/main.py -c 26 -s 1 -o 0.5 -a 30 -b base_portuguese_40
# python3 /src/tcc/main.py -c 18 -s 1 -o 0.5 -a 30 -b base_portuguese_40

# # 2 SEGUNDOS
# python3 /src/tcc/main.py -c 40 -s 2 -o 0.5 -a 30 -b base_portuguese_40
# python3 /src/tcc/main.py -c 34 -s 2 -o 0.5 -a 30 -b base_portuguese_40
# python3 /src/tcc/main.py -c 26 -s 2 -o 0.5 -a 30 -b base_portuguese_40
# python3 /src/tcc/main.py -c 18 -s 2 -o 0.5 -a 30 -b base_portuguese_40

# # 3 SEGUNDOS
# python3 /src/tcc/main.py -c 40 -s 3 -o 0.5 -a 30 -b base_portuguese_40
# python3 /src/tcc/main.py -c 34 -s 3 -o 0.5 -a 30 -b base_portuguese_40
# python3 /src/tcc/main.py -c 26 -s 3 -o 0.5 -a 30 -b base_portuguese_40
# python3 /src/tcc/main.py -c 18 -s 3 -o 0.5 -a 30 -b base_portuguese_40

# python3 /src/tcc/inference.py -m /src/tcc/models/base_portuguese/SEG_2_OVERLAP_0_AUG_30/MFCC_26/D60_DO0_D0_DO0_D0/1649191143_84.62809920310974

find /src/tcc/models -name inference -exec rm -rf {} \;

PARENT_FOLDER='/src/tcc/models/'

OUTPUT=$(python3 /src/tcc/backlog/run_folder.py -f $PARENT_FOLDER)

for i in $OUTPUT;
do
    python3 /src/tcc/inference.py -m "$PARENT_FOLDER$i" -i /src/tcc/dataset/inference
done