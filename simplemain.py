from pyhailing import RidehailEnv

def get_action(env, obs, noop: bool=False):
    if noop:
        return env.get_noop_action()
    else:
        return env.get_random_action()

def main():
    import matplotlib.pyplot as plt

    env = RidehailEnv(distances="manhattan_rotated")
    obs = env.reset()

    # rgb = env.render()
    # plt.imshow(rgb)

    terminal = False
    reward = 0

    while not terminal:

        action = get_action(env, obs)

        next_obs, new_rwd, terminal, _ = env.step(action)

        reward += new_rwd
        obs = next_obs

        # rgb = env.render()
        # plt.imshow(rgb)

    print(reward)
    print(obs)



if __name__ == "__main__":
    main()