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
from copy import deepcopy
from utils.check_room_count import RoomFinder, visualize_room, render_to_game, convert_to_layout, player_position
from utils.postprocessing import remove_unusable_parts
import reachability_modified_small

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Workspace(object):
    def __init__(self, args):
        self.args = args
        logger.info(self.args)
        self.corners = [(0, 0), (0, 4), (4, 0), (4, 4)]


        self.indices = [(i, j) for i in range(0, 5) for j in range(0, 5)]
        for i in self.indices:
            if i in self.corners:
                self.indices.remove(i)
        self.map_list = []
        self.map_list_test = []

        self.hamming_list = []
        self.hamming_list_test = []

        self.hamming_list = []

        self.result_dir = './results/room{0}_seed{1}'.format(self.args.room_count,self.args.seed)

        logger.info('Result path : {0}'.format(self.result_dir))
        random.seed(self.args.seed)


        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, 'images'), exist_ok=True)

        os.makedirs(os.path.join(self.result_dir, 'layouts'), exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, 'test_images'), exist_ok=True)

        os.makedirs(os.path.join(self.result_dir, 'test_layouts'), exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, 'test2_images'), exist_ok=True)

        os.makedirs(os.path.join(self.result_dir, 'test2_layouts'), exist_ok=True)


    def rand_position(self,num_items,layout,tmp_num):
        # random.choice 이후 sort list

        while True:
            tmp_break = True
            tmp_layout = deepcopy(layout)
            base_arr = np.array([2,3,4,5]).astype(int)
            while True:
                if tmp_num == 0:
                    item_arr =[2]
                else:
                    item_arr = np.random.choice(5, size=num_items-4, replace = True)

                counts = np.unique(item_arr, return_counts=True)[1]
                if max(counts) <2:
                    break
            if tmp_num==1:
                item_arr =item_arr+1
            new_arr=np.concatenate((base_arr,item_arr))
            sorted_item_list = np.sort(new_arr)
            selected_indices = np.random.choice(len(self.indices), size=num_items, replace=False)
            coordinates = [self.indices[i] for i in selected_indices]
            for i in range(num_items):
                x,y = coordinates[i]
                tmp_layout[x][y] = sorted_item_list[i]

            new_layout_0, removed_list_0 = remove_unusable_parts(tmp_layout)
            new_layout , removed_list = remove_unusable_parts(new_layout_0)

            final_removed_list = np.concatenate((removed_list_0,removed_list))
            for i in final_removed_list:
                indices = np.where(sorted_item_list == i)[0]
                sorted_item_list= np.delete(sorted_item_list, indices[0])
                if i not in sorted_item_list:
                    tmp_break =False
                    break
            if tmp_break == False:
                continue
            rooms = RoomFinder(new_layout).get_rooms()
            if len(rooms)!=1:
                continue

            if tmp_num==0:
                if len(new_layout[new_layout == 0]) < 7:
                    continue
            else:
                if len(new_layout[new_layout == 0]) < 9:
                    continue
            return new_layout
    def run(self):
        tmp_index = 0
        for k in range(100):
            individual = np.zeros((5, 5), dtype=int)

            for i in [(0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,4),(2,0),(2,4),(3,0),(3,4),(4,0),(4,1),(4,2),(4,3),(4,4)]:
                individual[i[0]][i[1]] = 1
            number = random.randint(6, 7)

            new_individual = self.rand_position(number,individual,1)


            if len(self.hamming_list) == 0:
                self.hamming_list.append(new_individual)
                tmp_index+=1
            else:
                if reachability_modified_small.input_or_not(self.hamming_list, new_individual)==1:
                    self.hamming_list.append(new_individual)
                    tmp_index += 1

        for j in range(len(self.hamming_list)-tmp_index,len(self.hamming_list)):

            hamming_map= self.hamming_list[j]
            hamming_map = player_position(hamming_map)
            if reachability_modified_small.get_solvability(hamming_map) == 1:
                self.map_list.append(hamming_map)
                print(len(self.map_list))
            if len(self.map_list) == 3050:#361
                break
    def run_testmap(self):
        tmp_index = 0
        for k in range(100):
            individual = np.zeros((5, 5), dtype=int)
            for i in [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 4), (2, 0), (2, 4), (3, 0), (3, 4), (4, 0),
                      (4, 1), (4, 2), (4, 3), (4, 4)]:
                individual[i[0]][i[1]] = 1
            number = 5
            new_individual = self.rand_position(number,individual,0)
            if len(self.hamming_list_test) == 0:
                self.hamming_list_test.append(new_individual)
                tmp_index+=1
            else:
                if reachability_modified_small.input_or_not(self.hamming_list_test, new_individual)==1:
                    self.hamming_list_test.append(new_individual)
                    tmp_index += 1
        for j in range(len(self.hamming_list_test)-tmp_index,len(self.hamming_list_test)):
            hamming_map= self.hamming_list_test[j]
            hamming_map = player_position(hamming_map)
            if reachability_modified_small.get_solvability(hamming_map) == 1:
                self.map_list_test.append(hamming_map)
                print(len(self.map_list_test))
            if len(self.map_list_test) == 50:#361
                break
    def main(self):
        while len(self.map_list) < 3050:  # 361
            self.run()

        while len(self.map_list_test) < 50:
            self.run_testmap()

        train_hamming_array, test_list, train_list = reachability_modified_small.build_hamming_list_4(self.map_list)

        appended_list = deepcopy(train_list)
        for i in range(50):  # 161
            appended_list.append(self.map_list_test[i])
        new_hamming_array = reachability_modified_small.build_hamminglist_3(appended_list)
        np.save(self.result_dir + '/hamming_distance array', train_hamming_array)
        np.save(self.result_dir + '/different rule array', new_hamming_array)  # x_save.npy
        index_1 = 0
        index_2 = 3000
        index_3 = 3050
        for i in range(len(train_list)):
                tmp_map = train_list[i]
                with open(os.path.join(self.result_dir, 'layouts', '{0}_processed.layout'.format(index_1)),
                          'w') as f:
                    layout = convert_to_layout(tmp_map.reshape(5, 5))
                    f.write(layout)
                player1 = [(i, j) for i in range(5) for j in range(5) if tmp_map[i][j] == 7][0]
                player2 = [(i, j) for i in range(5) for j in range(5) if tmp_map[i][j] == 8][0]

                tmp_map[player1[0], player1[1]] = 0
                tmp_map[player2[0], player2[1]] = 0

                new_game_image = render_to_game(tmp_map.reshape(5, 5)) # render to game 확인
                cv2.imwrite(os.path.join(self.result_dir, 'images', '{0}_processed.jpg'.format(index_1)),
                            new_game_image)
                index_1 += 1
        for i in range(len(test_list)):
                tmp_map = test_list[i]
                with open(os.path.join(self.result_dir, 'test_layouts', '{0}_processed.layout'.format(index_2)),
                          'w') as f:
                    layout = convert_to_layout(tmp_map.reshape(5, 5))
                    f.write(layout)
                player1 = [(i, j) for i in range(5) for j in range(5) if tmp_map[i][j] == 7][0]
                player2 = [(i, j) for i in range(5) for j in range(5) if tmp_map[i][j] == 8][0]

                tmp_map[player1[0], player1[1]] = 0
                tmp_map[player2[0], player2[1]] = 0

                new_game_image = render_to_game(tmp_map.reshape(5, 5)) # render to game 확인
                cv2.imwrite(os.path.join(self.result_dir, 'test_images', '{0}_processed.jpg'.format(index_2)),
                            new_game_image)
                index_2 += 1
        for i in range(len(self.map_list_test)):
                tmp_map = self.map_list_test[i]
                with open(os.path.join(self.result_dir, 'test2_layouts', '{0}_processed.layout'.format(index_3)),
                          'w') as f:
                    layout = convert_to_layout(tmp_map.reshape(5, 5))
                    f.write(layout)
                player1 = [(i, j) for i in range(5) for j in range(5) if tmp_map[i][j] == 7][0]
                player2 = [(i, j) for i in range(5) for j in range(5) if tmp_map[i][j] == 8][0]

                tmp_map[player1[0], player1[1]] = 0
                tmp_map[player2[0], player2[1]] = 0

                new_game_image = render_to_game(tmp_map.reshape(5, 5)) # render to game 확인
                cv2.imwrite(os.path.join(self.result_dir, 'test2_images', '{0}_processed.jpg'.format(index_3)),
                            new_game_image)
                index_3 += 1






if __name__ == '__main__':

    parser = argparse.ArgumentParser('Parameter settings for experiment')
    parser.add_argument('--room_count', type=int, required=True, help='Number of rooms in the map')

    parser.add_argument('--seed', type=int, required=True, help='Seed of the experiment')


    args = parser.parse_args()

    workspace = Workspace(args)
    workspace.main()
