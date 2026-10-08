"""Run closed-loop linear Gaussian MPC with REINFORCE."""

from mjmpc.cli import main

if __name__ == "__main__":
    main(default_controller="reinforce")
