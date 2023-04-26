#!/bin/bash

for i in {1..30}
do
  nohup python3 experiment.py --room_count 1 --pot_count 2 --onion_count 2 --dish_count 2 --outlet_count 2 --seed $i &
done