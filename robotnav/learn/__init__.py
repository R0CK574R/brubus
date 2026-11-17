"""Learning from the planner: the AI work on Brubus (README, "Learning from the planner").

Stage 1 (here now): the data. Hundreds of maps from the terrain generator, each labelled with the exact drive time
from every cell to a few goals, split by map, plus the baseline a learned estimate has to beat.
    labels.py    the labels and A* guided by a table of estimates (the Python reference; C++ in cpp/src/fields.cpp)
    dataset.py   building, splitting, storing and checking the data set
    baseline.py  how much A* searches with today's estimate, and with a perfect one
"""
