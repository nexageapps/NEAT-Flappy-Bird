# NEAT Flappy Bird

An AI that learns to play Flappy Bird using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm.

## Understanding NEAT - A Student's Guide (MAI UoA)

### What is NEAT?

**NEAT** = **NeuroEvolution of Augmenting Topologies**

It's an evolutionary algorithm that evolves neural networks - essentially "breeding" AI brains to solve problems without being explicitly programmed.

### Why NEAT Matters (MAI Context)

In Module AI (University of Auckland), NEAT demonstrates key concepts:

1. **Evolutionary Computation** - Mimics natural selection to solve optimization problems
2. **Neuroevolution** - Evolving neural network weights AND architectures
3. **Emergent Behavior** - Complex behavior emerges from simple fitness rules
4. **No Gradient Required** - Works where backpropagation fails (non-differentiable problems)

### How NEAT Differs from Standard Neural Networks

| Standard NN | NEAT |
|-------------|------|
| Fixed architecture | Evolves architecture (adds nodes/connections) |
| Trained via backpropagation | Trained via genetic algorithm |
| Needs labeled data | Works with fitness functions only |
| Local optimization | Global search with population |

### The Core Idea: Augmenting Topologies

NEAT starts with simple networks (input → output directly) and gradually adds complexity:

```
Generation 1:  [Input] → [Output]          (1 connection)
Generation 5:  [Input] → [Hidden] → [Output]  (2 connections)
Generation 10: [Input] → [Hidden1] → [Output]
                     ↓
                   [Hidden2] → (3 connections)
```

This "augmentation" helps NEAT find good solutions faster while avoiding premature convergence.

### The NEAT Algorithm (Step by Step)

```
1. INITIALIZE POPULATION
   └─ Create 50 random neural networks (different structures)
   
2. EVALUATE FITNESS
   └─ Test each network on the task
   └─ Score based on performance (e.g., distance traveled)
   
3. SELECT PARENTS
   └─ Higher fitness = higher chance to reproduce
   └─ Fitness sharing: protect niche species
   
4. REPRODUCE WITH MUTATION
   └─ Crossover: Combine two parents' networks
   └─ Mutation: Add node, add connection, change weight
   
5. REPEAT
   └─ Go back to step 2 for next generation
```

### Key NEAT Innovations

#### 1. **Genome Encoding**
Each network has a genome with:
- **Node genes**: Input, hidden, output neurons
- **Connection genes**: Weights and whether connection is enabled

#### 2. **Speciation**
Groups similar genomes to protect innovation:
- Prevents dominant species from taking over too soon
- Allows niche adaptation
- Controlled by `compatibility_threshold` in config

#### 3. **Complexity Control**
- Starts simple (avoids random complexity)
- Adds nodes/connections gradually
- Encourages minimal sufficient solutions

### Your Flappy Bird Example

#### Inputs (3 neurons)
```
1. Bird Y position          → How high the bird is
2. Distance to top pipe     → How far to obstacle
3. Distance to bottom pipe  → How far to ground
```

#### Output (1 neuron)
```
1. Jump probability         → Output > 0.5 = JUMP
```

#### Network Architecture (Evolves over generations)
```
Initial:                    After 10 generations:
[Input1] ────────┐           [Input1] ──→ [Hidden1] ──→ [Output]
                 ├─→ [Output]                ↓
[Input2] ────────┘                       [Input2] ────────────→
                                                 ↓
                                            [Input3] ────────────→
```

### Fitness Function Design (Critical!)

Your bird's fitness:
```python
+0.1  for each frame survived
+5.0  for each pipe passed
-1.0  for collision (in some implementations)
```

**Design principles:**
- **Shaping**: Small rewards guide behavior toward goal
- **No penalty for dying**: Encourages exploration
- **Bonus for pipes**: Rewards actual task completion

### Config File Breakdown

```ini
[NEAT]
pop_size = 50              # Population size (trade-off: speed vs diversity)
fitness_threshold = 100    # Stop when fitness reached (task complete)
no_fitness_termination = False  # Stop at threshold or max generations

[DefaultGenome]
num_inputs = 3             # Your 3 inputs (bird_y, top_dist, bot_dist)
num_outputs = 1            # Your 1 output (jump decision)
feed_forward = True        # No recurrent connections (simpler)
```

### Mutation Types

1. **Weight mutation** (80%): Adjust connection weight
   - `weight_mutate_power` controls how much changes

2. **Connection mutation** (5%): Add new connection
   - `conn_add_prob` = 0.5 chance per generation

3. **Node mutation** (20%): Split existing connection with new hidden node
   - `node_add_prob` = 0.2 chance per generation

### Visualizing Evolution

Run `visualize.py` to see:
- **avg_fitness.svg**: How average performance improves
- **speciation.svg**: Species diversity over time
- **draw_net()**: Visualize the evolved neural network

### Why NEAT is Cool for AI Research

1. **Black box problems**: Works when you can't compute gradients
2. **Discrete decisions**: Perfect for control problems
3. **Emergent behavior**: Complex strategies self-organize
4. **Biologically inspired**: Connects to cognitive science

### Limitations

- Slower than gradient descent for large networks
- No guarantee of global optimum
- Can overfit to specific task
- Hyperparameter sensitive (needs tuning)

### Further Reading (MAI Resources)

- **Paper**: [Evolving Neural Networks through Augmenting Topologies](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- **neat-python docs**: https://neat-python.readthedocs.io/
- **Stanford CS231n**: Evolutionary algorithms section
- **Your MAI lectures**: Check slides on evolutionary computation

## What is NEAT?

NEAT is a neuroevolution algorithm that evolves neural networks over generations. It starts with simple networks and gradually adds complexity (new nodes and connections) while training, finding optimal architectures for the given task.

## How It Works

1. Population: Starts with 50 random birds, each with a unique neural network
2. Evaluation: Each bird plays the game; fitness increases for surviving longer and passing pipes
3. Selection: Birds with higher fitness are more likely to reproduce
4. Evolution: New generations are created through mutation and crossover
5. Improvement: Over time, the AI learns to navigate through pipes effectively

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

1. Generation 0: 50 random birds with simple neural networks
2. Fitness Evaluation: Birds get +0.1 for each frame they survive, +5 for passing a pipe
3. Evolution: Better performing birds pass their genes to the next generation
4. Progress: Watch the AI improve over 50 generations

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

- best.pickle: Stores the best trained neural network
- avg_fitness.svg: Plot of average/best fitness over generations
- speciation.svg: Visualization of species distribution

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

Watch the detailed explanation

## Author

Original code by Tech With Tim

## License

This project is for educational purposes.
