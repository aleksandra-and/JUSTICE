#!/bin/bash

REMOTE="aandrasz@login.delftblue.tudelft.nl:/scratch/aandrasz/JUSTICE/results/continous_3_fixed/harl_justice_momarl/harl_justice_momarl/hasac"
DEST="./results/algo_comparison_v3_fixed/seeds/hasac"

mkdir -p "$DEST"

# Seed 1 - April 2026
rsync -av "$REMOTE/momarl_hasac_uniform_202604_w47_8690"  "$DEST/"  # weight 47

# Seed 2 - May 2026
rsync -av "$REMOTE/momarl_hasac_uniform_202605_w61_7016"  "$DEST/"  # weight 61
rsync -av "$REMOTE/momarl_hasac_uniform_202605_w59_4245"  "$DEST/"  # weight 59

# Seed 4 - May 2026
rsync -av "$REMOTE/momarl_hasac_uniform_202605_w20_9579"  "$DEST/"  # weight 20
rsync -av "$REMOTE/momarl_hasac_uniform_202605_w54_2483"  "$DEST/"  # weight 54
rsync -av "$REMOTE/momarl_hasac_uniform_202605_w49_2478"  "$DEST/"  # weight 49
rsync -av "$REMOTE/momarl_hasac_uniform_202605_w98_164"   "$DEST/"  # weight 98
