#!/bin/bash

REMOTE="aandrasz@login.delftblue.tudelft.nl:/scratch/aandrasz/JUSTICE/results/continous_3_fixed/harl_justice_momarl/harl_justice_momarl/mappo"
DEST="./results/algo_comparison_v3_fixed/seeds/mappo"

mkdir -p "$DEST"

# Seed 1 - April 2026
rsync -av "$REMOTE/momarl_mappo_uniform_202604_w73_164"   "$DEST/"  # weight 73
rsync -av "$REMOTE/momarl_mappo_uniform_202604_w93_8106"  "$DEST/"  # weight 93
rsync -av "$REMOTE/momarl_mappo_uniform_202604_w95_8040"  "$DEST/"  # weight 95
rsync -av "$REMOTE/momarl_mappo_uniform_202604_w98_1893"  "$DEST/"  # weight 98
rsync -av "$REMOTE/momarl_mappo_uniform_202604_w88_2250"  "$DEST/"  # weight 88
rsync -av "$REMOTE/momarl_mappo_uniform_202604_w58_6724"  "$DEST/"  # weight 58

# Seed 3 - May 2026
rsync -av "$REMOTE/momarl_mappo_uniform_202605_w32_8242"  "$DEST/"  # weight 32

# Seed 4 - May 2026
rsync -av "$REMOTE/momarl_mappo_uniform_202605_w42_4673"  "$DEST/"  # weight 42
rsync -av "$REMOTE/momarl_mappo_uniform_202605_w38_6437"  "$DEST/"  # weight 38
rsync -av "$REMOTE/momarl_mappo_uniform_202605_w29_6560"  "$DEST/"  # weight 29
rsync -av "$REMOTE/momarl_mappo_uniform_202605_w99_277"   "$DEST/"  # weight 100
