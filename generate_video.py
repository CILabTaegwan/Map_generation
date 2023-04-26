import cv2
import random
import numpy as np
import statistics
import argparse
import logging
import os
import json

from GA import GA
from utils.check_room_count import RoomFinder, visualize_room, render_to_game
from utils.postprocessing import remove_unusable_parts
from utils.image_preprocess import crop_and_resize, put_image, put_text
from initializer import initialize
from fitness_function import fitness_function

from recorder import VideoRecorder

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger(__name__)

font = cv2.FONT_HERSHEY_SIMPLEX


class Workspace(object):
    def __init__(self, args):
        self.args = args
        logger.info(self.args)

        self.result_dir = './results/room{0}_pot{1}_onion{2}_dish{3}_outlet{4}_seed{5}'.format(self.args.room_count,
                                                                                               self.args.pot_count,
                                                                                               self.args.onion_count,
                                                                                               self.args.dish_count,
                                                                                               self.args.outlet_count,
                                                                                               self.args.seed
                                                                                               )
        logger.info('Result path : {0}'.format(self.result_dir))
        random.seed(self.args.seed)

        # Save parameters to result directory
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, 'images'), exist_ok=True)

        with open(os.path.join(self.result_dir, 'parameters.json'), 'w') as f:
            json.dump(vars(self.args), f)

        self.best_fitness = None

        self.recorder = VideoRecorder()

    def run(self):

        pop1 = initialize()

        ga = GA(
            population=pop1,
            selection_method='roulette',
            fitness_func=fitness_function,
            args=self.args
        )

        for i in range(500):

            frame = np.zeros((1080, 1920, 3), dtype=float)

            frame = put_text(frame, f'Generation #{i + 1}', size=50, x=130, y=100)

            frame = put_text(frame, f'Experimental Parameters', size=30, x=130, y=780)
            frame = put_text(frame, f'Room Count (C): {args.room_count}', size=25, x=130, y=830)
            frame = put_text(frame, f'Pot Count (B_2): {args.pot_count}', size=25, x=130, y=870)
            frame = put_text(frame, f'Onion Count (B_3): {args.onion_count}', size=25, x=130, y=910)
            frame = put_text(frame, f'Dish Count (B_4): {args.onion_count}', size=25, x=130, y=950)

            ga.evolution()
            individual = ga.get_best_individual()

            frame = put_text(frame, f'Best Chromosome', size=30, x=330, y=220)
            best_fitness = ga.get_best_fitness()
            frame = put_text(frame, '(Fitness: {0:.3f})'.format(best_fitness), size=30, x=350, y=260)

            individual = np.array(list(individual)).astype(int)
            individual = individual.reshape(7, 7)
            rooms = RoomFinder(individual).get_rooms()

            visualize_room(individual.reshape(7, 7), rooms)
            game_image = render_to_game(individual.reshape(7, 7))

            # new_individual = remove_unusable_parts(individual)
            new_game_image = render_to_game(individual.reshape(7, 7))

            new_game_image = new_game_image[60:350, 45:483, :]
            new_game_image = cv2.resize(new_game_image, (400, 400))
            frame = put_image(new_game_image, frame, 250, 320)

            frame = put_text(frame, f'Population (16 of 100)', size=30, x=1180, y=100)

            for indiv_i in range(16):
                cand_individual_i = ga.population[ga.get_nth_best_index(indiv_i + 1)]
                cand_individual_i = np.array(list(cand_individual_i)).astype(int)
                cand_game_image_i = render_to_game(cand_individual_i.reshape(7, 7))
                cand_game_image_i = crop_and_resize(cand_game_image_i)

                y_idx, x_idx = indiv_i // 4, indiv_i % 4
                x_offset = 900 + (cand_game_image_i.shape[1] + 20) * x_idx
                y_offset = 150 + (cand_game_image_i.shape[0] + 20) * y_idx

                frame = put_image(cand_game_image_i, frame, x_offset, y_offset)

            # Record Here!!
            self.recorder.record(frame)

            new_individual = remove_unusable_parts(individual)
            new_game_image = render_to_game(new_individual.reshape(7, 7))
            new_game_image = new_game_image[60:350, 45:483, :]
            new_game_image = cv2.resize(new_game_image, (400, 400))

            frame = put_text(frame, '[Fixed]', size=25, x=420, y=730, rgba=(0, 0, 200, 0))
            frame = put_image(new_game_image, frame, 250, 320)

            # Record Here!!
            self.recorder.record(frame)

            fitness, detail = fitness_function(individual, self.args)

            cv2.imwrite(os.path.join(self.result_dir, 'images', '{0}_original.jpg'.format(i)), game_image)
            cv2.imwrite(os.path.join(self.result_dir, 'images', '{0}_processed.jpg'.format(i)), new_game_image)

            if self.best_fitness is None or self.best_fitness < fitness:
                cv2.imwrite(os.path.join(self.result_dir, 'best_original.jpg'.format(i)), game_image)
                cv2.imwrite(os.path.join(self.result_dir, 'best_processed.jpg'.format(i)), new_game_image)

                sizes = [len(rooms[room]) for room in rooms]

                if len(sizes) > 1:
                    sizes = np.array(sizes)
                    size_stdev = statistics.stdev(sizes)
                    fit_size_std = size_stdev
                else:
                    fit_size_std = 1

                # Count for each block, count rooms, record fitness value -> to dataframe pickle
                information = {
                    'fitness': fitness,
                    'room_count': len(rooms),
                    'room_size': fit_size_std,
                    'block_0': np.count_nonzero(new_individual == 0),
                    'block_1': np.count_nonzero(new_individual == 1),
                    'block_2': np.count_nonzero(new_individual == 2),
                    'block_3': np.count_nonzero(new_individual == 3),
                    'block_4': np.count_nonzero(new_individual == 4),
                    'block_5': np.count_nonzero(new_individual == 5),
                }

                with open(os.path.join(self.result_dir, 'best_chromosome.json'), 'w') as f:
                    json.dump(information, f)

                self.best_fitness = fitness

            logger.info('Generation {0:05} | mean {1:.4} | best {2:.4}'.format(i, ga.get_average_fitness(), ga.get_best_fitness()))

            self.recorder.save(os.path.join(self.result_dir, 'video.mp4'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parameter settings for experiment')

    parser.add_argument('--room_count', type=int, required=True, help='Number of rooms in the map')
    parser.add_argument('--pot_count', type=int, required=True, help='Number of pots in the map')
    parser.add_argument('--onion_count', type=int, required=True, help='Number of onions in the map')
    parser.add_argument('--dish_count', type=int, required=True, help='Number of dish in the map')
    parser.add_argument('--outlet_count', type=int, required=True, help='Number of outlet in the map')
    parser.add_argument('--seed', type=int, required=True, help='Seed of the experiment')

    args = parser.parse_args()

    workspace = Workspace(args)
    workspace.run()
