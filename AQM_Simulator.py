import heapq
import random
import networkx as nx
import time as py_time
import matplotlib.pyplot as plt
import networkx.exception

class AQMAlgorithm:
    def __init__(self):
        self.statistics = {'packet_loss': 0}  # Initialize packet loss tracking

    def decide_packet_drop(self, queue_length):
        # This method decides if a packet should be dropped based on the current queue length.
        return queue_length > 1000

    def adjust_queue(self, queue):
        # This method attempts to adjust the queue based on its current length, possibly dropping packets.
        packet_dropped = False
        if self.decide_packet_drop(len(queue)):
            print("Dropping packet due to queue length. Current queue length:", len(queue))
            queue.pop(0)  # Drop the oldest packet (FIFO approach)
            packet_dropped = True
            self.statistics['packet_loss'] += 1
        return packet_dropped

class SimpleRED(AQMAlgorithm):
    def __init__(self, min_thresh=200, max_thresh=800, max_prob=0.01):
        super().__init__()  # Initialize base class
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        self.max_prob = max_prob

    def decide_packet_drop(self, queue_length):
        # Enhanced packet drop decision based on RED algorithm parameters.
        if queue_length < self.min_thresh:
            return False
        elif queue_length > self.max_thresh:
            return True
        else:
            # Calculate drop probability in the intermediate range.
            prob = (queue_length - self.min_thresh) / (self.max_thresh - self.min_thresh) * self.max_prob
            return random.random() < prob

class NetworkEvent:
    # A class representing network events, such as packet arrivals and departures.
    def __init__(self, event_kind, time_of_arrival, time_of_departure=None, origin=None, target=None, data_volume=0, packet_id=None):
        self.event_kind = event_kind
        self.time_of_arrival = time_of_arrival
        self.time_of_departure = time_of_departure
        self.origin = origin
        self.target = target
        self.data_volume = data_volume
        self.packet_id = packet_id

    def __lt__(self, other):
        # Comparison method for prioritizing events based on their arrival time.
        return self.time_of_arrival < other.time_of_arrival

class SimulationScheduler:
    def __init__(self, network=None, aqm_algorithm=None):
        self.events_queue = []
        self.current_time = 0
        self.network = network
        self.statistics = {'event_processing_time': [], 'packet_processing_time': {}, 'packets_delivered': 0, 'packet_loss': 0, 'latencies': []}
        self.aqm_algorithm = aqm_algorithm if aqm_algorithm else AQMAlgorithm()
        self.network_queue = []

    def add_event(self, event):
        heapq.heappush(self.events_queue, event)

    def fetch_next_event(self):
        if self.events_queue:
            return heapq.heappop(self.events_queue)
        return None

    def process_events(self):
        while self.events_queue:
            event_start_time = py_time.time()  # Start timing the event processing
            event = self.fetch_next_event()
            self.current_time = event.time_of_arrival
            self.handle_event(event)
            event_end_time = py_time.time()  # End timing the event processing
            event_processing_time = event_end_time - event_start_time  # Calculate processing time
            self.statistics['event_processing_time'].append(event_processing_time)  # Record processing time

    def handle_event(self, event):
        if event.event_kind == "arrival":
            packet_dropped = self.aqm_algorithm.adjust_queue(self.network_queue)
            if not packet_dropped:
                self.network_queue.append(event)
                self.statistics['packet_processing_time'][event.packet_id] = event.time_of_arrival
                if self.network:
                    self.network.simulate_packet_flow(event)
            else:
                self.statistics['packet_loss'] += 1
        elif event.event_kind == "departure":
            if event.packet_id in self.statistics['packet_processing_time']:
                arrival_time = self.statistics['packet_processing_time'].pop(event.packet_id)
                processing_time = event.time_of_arrival - arrival_time
                self.statistics['latencies'].append(processing_time)
                self.statistics['packets_delivered'] += 1

    def report_statistics(self):
        total_packet_loss = self.statistics['packet_loss'] + self.aqm_algorithm.statistics['packet_loss']
        average_processing_time = sum(self.statistics['event_processing_time']) / len(self.statistics['event_processing_time']) if self.statistics['event_processing_time'] else 0
        average_latency = sum(self.statistics['latencies']) / len(self.statistics['latencies']) if self.statistics['latencies'] else 0
        print(f"Average Event Processing Time: {average_processing_time:.6f} seconds")
        print(f"Average Packet Latency: {average_latency:.8f} seconds")
        print(f"Packets Delivered: {self.statistics['packets_delivered']}")
        print(f"Total Packet Loss: {total_packet_loss}")
        self.plot_graphs()

    def plot_graphs(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(self.statistics['latencies'], bins=20, color='blue')
        plt.title('Packet Latency Distribution')
        plt.xlabel('Latency (s)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

class DynamicNetwork:
    def __init__(self, scheduler):
        self.graph = None
        self.scheduler = scheduler

    def create_topology(self, model_type, **params):
        if model_type == "BarabasiAlbert":
            self.graph = nx.barabasi_albert_graph(params['nodes'], params['edges'])
        elif model_type == "Waxman":
            self.graph = nx.waxman_graph(params['nodes'], alpha=params['alpha'], beta=params['beta'])
        else:
            raise ValueError("Unsupported topology model")

    def simulate_packet_flow(self, event):
        try:
            path = nx.shortest_path(self.graph, source=event.origin, target=event.target)
            for _ in path[:-1]:  # Simulate the travel of the packet through the network
                event.time_of_arrival += 0.01  # Assume each hop adds a fixed delay
            departure_event = NetworkEvent("departure", event.time_of_arrival, None, event.origin, event.target, event.data_volume, event.packet_id)
            self.scheduler.add_event(departure_event)
        except nx.NetworkXNoPath:
            print(f"No path found between {event.origin} and {event.target}. Packet {event.packet_id} lost.")
            self.scheduler.statistics['packet_loss'] += 1

def generate_initial_events(scheduler, num_events, network):
    nodes = list(network.graph.nodes())
    for _ in range(num_events):
        time_of_arrival = random.uniform(0, 100)
        origin, target = random.sample(nodes, 2)
        data_volume = random.randint(64, 1500)  # Bytes
        packet_id = random.randint(1, 10000)
        event = NetworkEvent("arrival", time_of_arrival, None, origin, target, data_volume, packet_id)
        scheduler.add_event(event)

def main():
    print("Initializing AQM strategy and simulation scheduler")
    aqm_strategy = SimpleRED(min_thresh=300, max_thresh=700, max_prob=0.1)
    scheduler = SimulationScheduler(aqm_algorithm=aqm_strategy)

    print("Creating network and generating initial events")
    network = DynamicNetwork(scheduler)
    scheduler.network = network
    network.create_topology("BarabasiAlbert", nodes=100, edges=2)
    generate_initial_events(scheduler, 1000, network)

    print("Processing events")
    scheduler.process_events()

    print("Reporting simulation statistics:")
    scheduler.report_statistics()

if __name__ == "__main__":
    main()

