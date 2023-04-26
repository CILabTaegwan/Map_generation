#!/bin/bash

for i in {1..100}
do
  nohup python3 experiment.py --room_count 2 --pot_count 3 --onion_count 3 --dish_count 3 --outlet_count 1 --seed $i &
  nohup python3 experiment.py --room_count 2 --pot_count 3 --onion_count 7 --dish_count 5 --outlet_count 1 --seed $i &
  nohup python3 experiment.py --room_count 2 --pot_count 7 --onion_count 7 --dish_count 7 --outlet_count 1 --seed $i &
done

for i in {1..100}
do
  # nohup python3 experiment.py --room_count 1 --pot_count 3 --onion_count 3 --dish_count 3 --outlet_count 1 --seed $i &
  # nohup python3 experiment.py --room_count 2 --pot_count 3 --onion_count 3 --dish_count 3 --outlet_count 1 --seed $i &
  # nohup python3 experiment.py --room_count 3 --pot_count 3 --onion_count 3 --dish_count 3 --outlet_count 1 --seed $i &
done