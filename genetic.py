import pandas as pd
from sklearn import datasets
import numpy as np
import itertools
 
iris = datasets.load_iris()
 
df = pd.DataFrame(iris.data, columns=iris.feature_names)


class Chromosome:
  data_frame = df
  def __init__(self, chromosome):
    self.chromosome = chromosome
    self.centeroids = self.find_centeroids()
    self.fitness = self.evaluation_function()

  def find_centeroids(self):
    centeroids = []
    for i in range(self.chromosome[150]):
      centeroid_of_cluster = [0, 0, 0, 0]
      centeroids.append(centeroid_of_cluster)

    for i in range(150):
      for j in range(4):
        centeroids[self.chromosome[i]][j] += self.data_frame.iloc[i, j]

    for i in range(int(self.chromosome[150])):
      centeroids[i] = np.divide(centeroids[i], np.count_nonzero(self.chromosome == i))
    
    return centeroids

  
  def evaluation_function(self):
    fitness = 0
    for i in range(150):
      for j in range(4):
        fitness += abs(self.centeroids[self.chromosome[i]][j] - self.data_frame.iloc[i, j])

    return fitness
  

  def mutation(self):
    mutation_chromosome = np.copy(self.chromosome)
    for i in range(150):
      for j in range(self.chromosome[150]):
        if(sum(np.abs(self.data_frame.iloc[i, :] - self.centeroids[self.chromosome[i]])) > sum(np.abs(self.data_frame.iloc[i, :] - self.centeroids[j]))):
          mutation_chromosome[i] = j

    return mutation_chromosome
  

  def accuracy(self):
    target = iris.target
    max_score = 0
    lists = [
      [0, 1, 2],
      [0, 2, 1],
      [1, 0, 2],
      [1, 2, 0],
      [2, 0, 1],
      [2, 1, 0]]
    for p in lists:
      score = 0
      for i in range(150):
        if(self.chromosome[i] == p[target[i]]):
          score += 1
      
      if score > max_score:
        max_score = score

    return max_score / 150
  







class Genetic:

  def __init__(self, number_of_clusters: int, population: int, maximum_number_of_generatins: int, surviver_percentage: float, number_of_mutation: int):
    self.df = df
    self.number_of_clusters = number_of_clusters
    self.population = population
    self.number_of_mutations = number_of_mutation
    self.surviver_percentage = surviver_percentage
    self.chromosomes = self.first_generation()
    self.sort()
    for i in range(maximum_number_of_generatins):
      self.next_generation()
      self.sort()
      for j in range(self.number_of_mutations):
        self.chromosomes.pop()

      print(i)
      print(self.chromosomes[0].fitness)



  def first_generation(self):
    chromosomes = []
    for i in range(self.population):
      temp_array = np.random.randint(self.number_of_clusters, size=(150))
      temp_array = np.append(temp_array, self.number_of_clusters)
      chromosome = Chromosome(temp_array)
      chromosomes.append(chromosome)

    return chromosomes
    

  def sort(self):
    for i in range(len(self.chromosomes)):
      for j in range(i - 1, -1, -1):
        if(self.chromosomes[j].fitness > self.chromosomes[j + 1].fitness):
          temp_chromosome = self.chromosomes[j]
          self.chromosomes[j] = self.chromosomes[j + 1]
          self.chromosomes[j + 1] = temp_chromosome
        else: break


  def next_generation(self):
    number_of_survivers = int(self.population * self.surviver_percentage)
    k = number_of_survivers
    over = False
    for i in range(number_of_survivers):
      for j in range(i + 1, number_of_survivers):
        crossover_chromosomes = self.crossover(self.chromosomes[i], self.chromosomes[j])
        self.chromosomes[k] = crossover_chromosomes[0]
        k = k + 1
        if(k >= self.population):
          over = True
          break
    
      if(over): break

    for i in range(self.number_of_mutations):
      rand = np.random.randint(10)
      chromosome = Chromosome(self.chromosomes[rand].mutation())
      self.chromosomes.append(chromosome)

    


  def crossover(self, chromosome1, chromosome2):
    x = chromosome1.chromosome
    y = chromosome2.chromosome
    rand = np.random.randint(50, 100)
    for i in range(rand):
      temp = x[i]
      x[i] = y[i]
      y[i] = temp

    x_chromosome = Chromosome(x)
    y_chromosome = Chromosome(y)

    return(x_chromosome, y_chromosome)
  
