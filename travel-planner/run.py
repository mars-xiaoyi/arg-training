import os
import json
from agent import PreferenceAgent


def main():
    """
    Main function to run the travel agent.
    """
    agent = PreferenceAgent()
    completion_message = agent.run()
    print(completion_message)


if __name__ == "__main__":
    main()