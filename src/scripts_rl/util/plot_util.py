import numpy as np
import matplotlib.pyplot as plt

def plot_actionHistory(agent_actions, plot_dir, episode):
    """Plot agent actions with fading colors and save the plot."""
    fig, ax = plt.subplots()
    num_agent_actions = len(agent_actions)
    agent_colors = plt.cm.Blues(np.linspace(0.3, 1, num_agent_actions))

    for i, action in enumerate(agent_actions):
        ax.scatter(action[0], action[1], color=agent_colors[i], s=10, label="Agent Action" if i == 0 else "")

    ax.set_xlabel("Action X")
    ax.set_ylabel("Action Y")
    ax.set_title("Agent Actions Over Time")
    ax.legend()

    # Save the plot image
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/agent_actions_{episode}.png")
    plt.close()


def plot_rewards_epsilons(rewards, epsilons, episode, plot_dir):
    # Create the figure and the first y-axis
    fig, ax1 = plt.subplots()

    # Plot the rewards on the first y-axis
    ax1.plot(rewards, label="Reward", color="tab:blue")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Create the second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    # Plot the epsilons on the second y-axis
    ax2.plot(epsilons, label="Epsilon", color="tab:orange")
    ax2.set_ylabel("Epsilon", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Add the legend
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Save the plot image
    plt.tight_layout()  # Avoid cutting off labels
    plt.savefig(f"{plot_dir}/dqn_rewards_epsilons_{episode}.png")
    plt.close()

    # Save the rewards and epsilons to a CSV file
    data = np.column_stack((rewards, epsilons))
    np.savetxt(f"{plot_dir}/dqn_rewards_epsilons_{episode}.csv", data, delimiter=",", header="Reward,Epsilon", comments="")


def plot_losses_epsilons(losses, epsilons, episode, plot_dir):
    # Create the figure and the first y-axis
    fig, ax1 = plt.subplots()

    # Plot the losses on the first y-axis
    ax1.plot(losses, label="Loss", color="tab:blue")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Create the second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    # Plot the epsilons on the second y-axis
    ax2.plot(epsilons, label="Epsilon", color="tab:orange")
    ax2.set_ylabel("Epsilon", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Add the legend
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")    

    # Save the plot image
    plt.tight_layout()  # Avoid cutting off labels
    plt.savefig(f"{plot_dir}/dqn_losses_epsilons_{episode}.png")
    plt.close()

    # Save the losses and epsilons to a CSV file
    data = np.column_stack((losses, epsilons))
    np.savetxt(f"{plot_dir}/dqn_losses_epsilons_{episode}.csv", data, delimiter=",", header="Losses,Epsilon", comments="")
