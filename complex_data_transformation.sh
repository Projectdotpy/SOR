#!/usr/bin/env bash

cd data

shopt -s globstar
unzip instre_monuments.zip
cd instre_monuments
for i in **/*{.txt,.jpg}; do mv "$i" "${i//\//_}"; done
rm -r */

cd ../..

./create_csv.py
