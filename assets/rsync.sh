#!/bin/sh

# devfair_dir=/checkpoint/bbolte/eval/today/language-navigation
devfair_dir=/private/home/bbolte/Experiments/Evaluation/today/language-navigation/eval-20230301-132436

# Command to copy files from remote machine to local machine.
rsync -r "devfair:${devfair_dir}/*" .
