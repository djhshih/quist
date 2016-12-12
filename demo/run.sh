#!/bin/bash

Rscript generate.R

mkdir -p output

quist \
	--window 5 \
	--output-raw output/raw-scores.vtr \
	--output-modes output/modes.tsv \
	input.tsv output/scores.vtr

Rscript analyze.R

