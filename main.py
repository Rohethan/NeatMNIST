import multiprocessing
import random

import neat
import numpy
from tqdm import tqdm
import numba

print("loading dataset")
dataset = numpy.load("mnist-5000-normalized.npy")


def get_random_digit_data():
	random_digit_class = random.randrange(0, 10)
	random_digit_data_index = random.randrange(0, 5000)
	return random_digit_class, dataset[random_digit_class][random_digit_data_index]


def build_evalset():
	evaluation_set = []
	for i in range(50):
		evaluation_set += [get_random_digit_data()]
	return evaluation_set


def eval_genome(genomes, config):
	for genome_id, genome in tqdm(genomes):
		genome.fitness = 0
		evaluation_set = build_evalset()
		net = neat.nn.FeedForwardNetwork.create(genome, config)
		for i in range(len(evaluation_set)):
			net_out = net.activate(evaluation_set[i][1])
			if net_out.index(max(net_out)) == evaluation_set[i][0]:
				genome.fitness += 1
		#print(genome.fitness)

print("building config")
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
					 neat.DefaultSpeciesSet, neat.DefaultStagnation,
					 "config_feedforward")

p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(5))
print("training start")
winner = p.run(eval_genome, 300)
