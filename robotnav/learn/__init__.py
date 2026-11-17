"""Learning from the planner: the AI work on Brubus (README, "Learning from the planner").

Stage 1, the data: hundreds of maps from the terrain generator, each labelled with the exact drive time from every
cell to a few goals, split by map, plus the baseline a learned estimate has to beat.
    labels.py    the labels and A* guided by a table of estimates (the Python reference; C++ in cpp/src/fields.cpp)
    dataset.py   building, splitting, storing and checking the data set
    baseline.py  how much A* searches with today's estimate, and with a perfect one

Stage 2, a learned estimate for A*, benchmarked against the classical ones (landmarks won: robotnav/landmarks.py).
    features.py  what the network reads about a cell and its goal
    model.py     the network's file and its inference (C++ in cpp/src/learned.cpp, bit for bit)
    train.py     training it with PyTorch
    evaluate.py  the benchmark

Stage 3, Bruce learns his own terrain speeds from his driving, in a simulated world whose true speeds he can't see.
    speeds.py    the hidden true world, the driving logs, the network of seconds per metre, and the study
"""
