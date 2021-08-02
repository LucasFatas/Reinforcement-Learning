import random


class MyEGreedy:

    def __init__(self):
        print("Made EGreedy")

    def get_random_action(self, agent, maze):
        valid_actions = maze.get_valid_actions(agent)  # Gets all possible actions for this agent
        random_action = random.choice(valid_actions)  # Selects a random action from the list of possible actions
        return random_action

    def get_best_action(self, agent, maze, q_learning):
        valid_actions = maze.get_valid_actions(agent)  # Gets all possible actions for this agent
        action_values = q_learning.get_action_values(agent.get_state(maze), valid_actions)
        if sum(action_values) == 0:
            return self.get_random_action(agent, maze)

        max_index = action_values.index(max(action_values))
        return valid_actions[max_index]

    def get_egreedy_action(self, agent, maze, q_learning, epsilon):
        if random.random() < 1 - epsilon:
            return self.get_best_action(agent, maze, q_learning)
        else:
            return self.get_random_action(agent, maze)
