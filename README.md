# Network-Traffic-Management-Simulator
A Python-based simulation framework for evaluating and comparing Active Queue Management (AQM) algorithms within dynamic network topologies.

This project introduces a simulation framework aimed at exploring and evaluating Active Queue Management (AQM) algorithms within a networked environment. The focus is on enhancing network performance by implementing and comparing different AQM strategies, specifically targeting improvements over the traditional First In, First Out (FIFO) method.

# Features
Implementation of a base AQM algorithm class for packet loss tracking and queue adjustment based on queue length.
A SimpleRED (Random Early Detection) AQM algorithm that extends the base class with configurable parameters for minimum and maximum thresholds and maximum drop probability.
Simulation of network events (packet arrivals and departures) with prioritization based on arrival times.
A dynamic network topology setup using the NetworkX library to model complex network scenarios.
Comprehensive event processing within the network to simulate real-world traffic management and packet flow.
Statistical analysis of simulation outcomes, including packet loss, event processing time, and packet latency distribution.

# Dependencies
Python 3.x

NetworkX: For network topology creation and manipulation.

Matplotlib: For generating plots of simulation statistics.

# Implementation Details
The core components of the simulation include:

1. AQMAlgorithm and SimpleRED: Classes defining AQM strategies.
2. NetworkEvent: A class for representing network events.
3. SimulationScheduler: Manages the event queue and processes events based on the simulated network time.
4.  DynamicNetwork: Handles network topology and simulates packet flow within the network.

# Future Work
Future iterations of this project could explore the integration of machine learning models to predict and manage congestion more effectively, the addition of more complex network topologies, and the implementation of other AQM algorithms for comparative analysis.
