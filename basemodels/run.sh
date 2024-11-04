#!/bin/bash

rm ./logs/*
rm -rf ./checkpoints/basemodel/*

python cluster.py

python args.py

echo "args end"

for var in {-1..2}
do 
  #nohup  python -u basemodel.py $var >> ./logs/$var.log & 
  python -u basemodel.py $var 
done

