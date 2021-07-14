import matplotlib.pyplot as plt

from pyhailing import RidehailEnv


def main(render:bool=False):

    all_eps_rewards = []
    
    env = RidehailEnv(**RidehailEnv.DIMACS_CONFIGS.SUI)

    for episode in range(RidehailEnv.DIMACS_NUM_EVAL_EPISODES):

        obs = env.reset()
        terminal = False
        reward = 0

        if render:
            rgb = env.render()
            plt.imshow(rgb)
            plt.show()

        while not terminal:

            action = env.get_noop_action()
            # action = env.get_random_action()

            next_obs, new_rwd, terminal, _ = env.step(action)

            reward += new_rwd
            obs = next_obs

            if render:
                rgb = env.render()
                plt.imshow(rgb)
                plt.show()

        print(f"Episode {episode} complete. Reward: {reward}")
        all_eps_rewards.append(reward)

    mean_reward = sum(all_eps_rewards)/len(all_eps_rewards)
    print(f"All episodes complete. Average reward: {mean_reward}")


if __name__ == "__main__":
    main()