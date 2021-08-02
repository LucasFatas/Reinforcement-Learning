from Assignment_3.main.QLearning import QLearning


class MyQLearning(QLearning):

    def update_q(self, state, action, r, state_next, possible_actions, alpha, gamma):
        old_q = self.get_q(state, action)
        max_next_q = max([self.get_q(state_next, p_act) for p_act in possible_actions])

        new_q = old_q + alpha * (r + gamma * max_next_q - old_q)

        self.set_q(state, action, new_q)
