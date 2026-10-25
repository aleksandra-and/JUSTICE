#!/bin/bash

REMOTE="aandrasz@login.delftblue.tudelft.nl:/scratch/aandrasz/JUSTICE/results/continous_3_fixed/harl_justice_momarl/harl_justice_momarl/happo"
DEST="./results/algo_comparison_v3_fixed/seeds/happo"

mkdir -p "$DEST"

# Seed 1 - April 2026
rsync -av "$REMOTE/momarl_happo_uniform_202604_w83_6756"  "$DEST/"  # weight 83
rsync -av "$REMOTE/momarl_happo_uniform_202604_w99_3674"  "$DEST/"  # weight 100

# Seed 3 - May 2026
rsync -av "$REMOTE/momarl_happo_uniform_202605_w46_6153"  "$DEST/"  # weight 46

# Seed 4 - May 2026
rsync -av "$REMOTE/momarl_happo_uniform_202605_w40_9654"  "$DEST/"  # weight 40
rsync -av "$REMOTE/momarl_happo_uniform_202605_w34_349"   "$DEST/"  # weight 34
rsync -av "$REMOTE/momarl_happo_uniform_202605_w60_4220"  "$DEST/"  # weight 60
rsync -av "$REMOTE/momarl_happo_uniform_202605_w45_8939"  "$DEST/"  # weight 45
rsync -av "$REMOTE/momarl_happo_uniform_202605_w37_591"   "$DEST/"  # weight 37
rsync -av "$REMOTE/momarl_happo_uniform_202605_w47_1150"  "$DEST/"  # weight 47
rsync -av "$REMOTE/momarl_happo_uniform_202605_w39_6570"  "$DEST/"  # weight 39
rsync -av "$REMOTE/momarl_happo_uniform_202605_w59_4637"  "$DEST/"  # weight 59
rsync -av "$REMOTE/momarl_happo_uniform_202605_w85_6768"  "$DEST/"  # weight 85
rsync -av "$REMOTE/momarl_happo_uniform_202605_w93_2776"  "$DEST/"  # weight 93
rsync -av "$REMOTE/momarl_happo_uniform_202605_w88_6328"  "$DEST/"  # weight 88
rsync -av "$REMOTE/momarl_happo_uniform_202605_w95_9187"  "$DEST/"  # weight 95
rsync -av "$REMOTE/momarl_happo_uniform_202605_w97_8825"  "$DEST/"  # weight 97
rsync -av "$REMOTE/momarl_happo_uniform_202605_w99_7550"  "$DEST/"  # weight 100
