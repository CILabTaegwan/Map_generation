import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
import statistics
import argparse
import logging
import os
import pandas as pd
import seaborn as sns
import json
import experiment
from multiprocessing import Pool
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class map_gen(object):
    def __init__(self,args):
        self.args= args
        pool_obj = Pool(processes=5)
        pool_obj.map(self.gen_room_1,range(0,5))
        pool_obj.map(self.gen_room_2,range(0,5))
    def gen_room_1(self,step):
        for i in range(8):
            args1 =self.args
            args1.room_count = 1
            args1.seed = i+8*step
            args1.pot_count = random.randint(1, 2)
            args1.onion_count = random.randint(1, 2)
            args1.dish_count = random.randint(1, 2)
            args1.outlet_count = random.randint(1, 2)
            args1.factor_between_distance = 0
            a = experiment.Workspace(args1)
            a.run()
    def gen_room_2(self,step):
        for i in range(8):
            args2 = self.args
            args2.room_count = 2
            args2.seed = i+ 8*step + 40
            args2.pot_count = random.randint(1, 2)
            args2.onion_count = random.randint(1, 2)
            args2.dish_count = random.randint(1, 2)
            args2.outlet_count = random.randint(1, 2)
            args2.factor_between_distance = 0
            b = experiment.Workspace(args2)
            b.run()
    def plus_1(self,a,b):
        return a+b
if __name__ == '__main__':

    parser = argparse.ArgumentParser('Parameter settings for experiment')
    parser.add_argument('--room_count', type=int, required=True, help='Number of rooms in the map')
    parser.add_argument('--pot_count', type=int, required=True, help='Number of pots in the map')
    parser.add_argument('--onion_count', type=int, required=True, help='Number of onions in the map')
    parser.add_argument('--dish_count', type=int, required=True, help='Number of dish in the map')
    parser.add_argument('--outlet_count', type=int, required=True, help='Number of outlet in the map')
    parser.add_argument('--factor_distance', type=float, default=0., help='Distance factor in the fitness function')
    parser.add_argument('--factor_between_distance', type=float, default=0., help='Distance factor in the fitness function')
    parser.add_argument('--seed', type=int, required=True, help='Seed of the experiment')


    args = parser.parse_args()


    a= map_gen(args)
    #workspace = Workspace(args)
    #workspace.run()
