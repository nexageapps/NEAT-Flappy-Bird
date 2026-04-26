# NEAT Flappy Bird 🐦

An AI that learns to play Flappy Bird using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm.

## What is NEAT?

NEAT is a neuroevolution algorithm that evolves neural networks over generations. It starts with simple networks and gradually adds complexity (new nodes and connections) while training, finding optimal architectures for the given task.

## How It Works

1. **Population**: Starts with 50 random birds, each with a unique neural network
2. **Evaluation**: Each bird plays the game; fitness increases for surviving longer and passing pipes
3. **Selection**: Birds with higher fitness are more likely to reproduce
4. **Evolution**: New generations are created through mutation and crossover
5. **Improvement**: Over time, the AI learns to navigate through pipes effectively

## Installation

### Requirements

- Python 3.7+
- pip package manager

### Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

Dependencies:
- `numpy` - Numerical computing
- `pygame` - Game rendering
- `neat-python` - NEAT algorithm implementation
- `graphviz` - Network visualization
- `matplotlib` - Fitness plotting

## Running the Project

### First Run Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the AI training
python flappy_bird.py
```

### How It Works

1. **Generation 0**: 50 random birds with simple neural networks
2. **Fitness Evaluation**: Birds get +0.1 for each frame they survive, +5 for passing a pipe
3. **Evolution**: Better performing birds pass their genes to the next generation
4. **Progress**: Watch the AI improve over 50 generations

### Running the Trained AI

After training completes, the best neural network is saved to `best.pickle`. To run with the trained model:

```bash
python flappy_bird.py
```

The AI will automatically load the best genome and play the game.

### Stopping the Simulation

- **Close the window** to stop training early
- The simulation runs for up to 50 generations by default

### Viewing Training Statistics

The terminal shows:
- **Population's average fitness**: Mean performance of the generation
- **Best fitness**: Top performer's score
- **Generation time**: How long each generation took

## Output Files

- `best.pickle`: Stores the best trained neural network
- `avg_fitness.svg`: Plot of average/best fitness over generations
- `speciation.svg`: Visualization of species distribution

## Configuration

Edit `config-feedforward.txt` to customize:
- Population size (`pop_size`)
- Fitness threshold (`fitness_threshold`)
- Mutation rates
- Network parameters (inputs: 3, outputs: 1)

## Inputs to Neural Network

1. Bird's Y position
2. Distance to top pipe
3. Distance to bottom pipe

## Output

Single neuron: Jump if output > 0.5

## Video Tutorial

[Watch the detailed explanation](https://www.youtube.com/watch?v=OGHA-elMrxI)

## Author

Original code by [Tech With Tim](https://techwithtim.net)

## License

This project is for educational purposes.
