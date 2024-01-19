import pandas as pd 
from genetic import Genetic
import numpy as np

 
genetic = Genetic(3, 100, 15, 0.2, 1)

print(genetic.chromosomes[0].chromosome)
print(genetic.chromosomes[0].accuracy())