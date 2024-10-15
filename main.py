import json
import sys
from math import inf

from tabulate import tabulate

import simulator


class Parameters:
    """Store simulation and queue parameters from a parsed JSON file."""

    def __init__(self, params):
        self.random_limit: int = params['random_limit']
        """The amount of random numbers to generate
        before ending the simulation.
        """
        self.pregen: bool = params['pregen']
        """If True, random numbers come from a pregenerated list."""

        # Store random number generation parameters
        # according to the generation method.
        if self.pregen:
            self.randoms: list[float] = params['randoms']
            """Random number list."""
        else:
            self.a: float = params['a']
            """Random number generation parameter."""
            self.c: float = params['c']
            """Random number generation parameter."""
            self.m: float = params['m']
            """Random number generation parameter."""
            self.seed: float = params['seed']
            """Random number generation parameter."""

        self.start_queue: str = params['start_queue']
        """Name of the queue which receives all arrival events."""
        self.start_time = params['start_time']
        """Time for first arrival."""

        self.arrival_range: range = range(
            params['arrival_range'][0], params['arrival_range'][1]
        )
        """Time range for how long it takes for an arrival to occur."""

        self.queues: list[simulator.Queue] = []
        """Queues in the simulation."""

        # Store each individual queue as a queue object.
        for q_name, q_params in params['queues'].items():
            departure_range = range(
                q_params['departure_range'][0], q_params['departure_range'][1]
            )

            max_queue_size = q_params['max_queue_size']

            if max_queue_size is None:
                max_queue_size = inf

            self.queues.append(simulator.Queue(
                q_name, q_params['servers'], max_queue_size,
                departure_range, q_params['out']
            ))


def main():
    """Program entrypoint."""
    simul, params = parse_params_file(sys.argv[1])
    run_simulation(simul, params)
    print_results(simul)


def parse_params_file(filename: str) -> tuple[simulator.Simulator, Parameters]:
    """Parse a JSON input file into a parameters object."""
    with open(filename) as fp:
        params = Parameters(json.load(fp))

    if params.pregen:
        rand = simulator.RandomFromList(params.randoms)
    else:
        rand = simulator.Random(params.a, params.c, params.m, params.seed)

    simul = simulator.Simulator(
        params.queues, params.start_queue, params.arrival_range, rand
    )

    return simul, params


def run_simulation(simul: simulator.Simulator, params: Parameters):
    """Step through the simulation until the stop condition is achieved."""
    simul.start(params.start_time)

    while (simul.random_generated < params.random_limit):
        simul.step()


def print_results(simul: simulator.Simulator):
    """Print global simulation and individual queue results"""
    print(f'Final time: {simul.time:.2f}')

    for q in simul.queues.values():
        print(f'\n{q.name}\n')

        table = []
        headers = ['Queue size', 'Time', 'Probability %']

        for i, time in enumerate(q.times_per_size):
            probability = (time / simul.time) * 100
            table.append([i, time, probability])

        print(tabulate(table, headers, floatfmt='.2f'))

        print(f'\nEvents lost: {q.events_lost}')


if __name__ == '__main__':
    main()
