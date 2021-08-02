from Assignment_3.main.Maze import Maze
from Assignment_3.main.Agent import Agent
from Assignment_3.main.mysolution.MyQLearning import MyQLearning
from Assignment_3.main.mysolution.MyEGreedy import MyEGreedy

from matplotlib import pyplot as plt

if __name__ == "__main__":
    num_runs = 10
    average_of_trials_per_run = []

    for _ in range(num_runs):
        # load the maze
        #file = "..\\..\\data\\toy_maze.txt"
        file = "C:/Users/mateja/Documents/ci-group-23/Assignment_3/data/toy_maze.txt"
        maze = Maze(file)

        # Set the reward at the bottom right to 10
        goal_state = maze.get_state(9, 9)
        maze.set_reward(goal_state, 10)

        # Set the second reward to top right, (9,0)
        goal_state2 = maze.get_state(9,0)
        maze.set_reward(goal_state2, 5)

        # create a robot at starting and reset location (0,0) (top left)
        robot = Agent(0, 0)

        # make a selection object (you need to implement the methods in this class)
        selection = MyEGreedy()

        # make a QLearning object (you need to implement the methods in this class)
        learn = MyQLearning()

        stop = False
        steps = 0
        stop_criterion = 30000

        epsilon = 0.1
        beg_epsilon = 0.1
        small_epsilon = 0.00005
        alpha = 0.7
        gamma = 0.001

        trials = []

        # keep learning until you decide to stop
        while not stop:
            prev_state = robot.get_state(maze)
            # Decrement value of epsilon with every trial by small_epsilon
            epsilon = beg_epsilon - (small_epsilon * sum(trials))
            action = selection.get_egreedy_action(robot, maze, learn, epsilon)
            next_state = robot.do_action(action, maze)
            r = maze.get_reward(next_state)
            learn.update_q(prev_state, action, r, next_state, maze.get_valid_actions(robot), alpha, gamma)

            if next_state == goal_state:
                trials.append(robot.nr_of_actions_since_reset)
                robot.reset()


            if next_state == goal_state2:
                trials.append(robot.nr_of_actions_since_reset)
                robot.reset()

            steps += 1
            if steps == stop_criterion:
                stop = True

        print("Optimal path length: ", min(trials))

        average = round(sum(trials) / len(trials))
        print("Average of trials : ", average)
        average_of_trials_per_run.append(trials)

    length = max([len(t) for t in average_of_trials_per_run])
    result = [sum([r[t - 1] if len(r) >= t else 0 for r in average_of_trials_per_run]) / sum([1 if len(r) >= t else 0 for r in average_of_trials_per_run]) for t
              in range(1, length + 1)]

    plt.title("Average Number Steps per Trial")
    plt.plot(result)
    plt.ylabel("Average Steps")
    plt.xlabel("Trial")
    plt.show()
