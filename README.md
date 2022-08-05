# Deep Q-Learning PVP AI

This project has been abandoned in favor of a better simulation environment in [pogo-simulator](https://github.com/benggarza/pogo-simulator) designed for the sole reason of developing AI models

## Basic Workflow

1. User sets up a battle managed through TrainingSetupInterface and MatchHandler, through which a new Battle, Player, PlayerAI, and PlayerModel are created. ** previous Q-function and perceptron model are loaded from a json file **
2. Battle begins, updates the battle state in step() at 0.5 second intervals
3. At each step(), PlayerAI asks PlayerModel to push previous state, action, and reward to PlayerMemory queue, and call the perceptron network to predict an action that will yield the largest reward (approximating the Q-function). Rewards are awarded if an opponent pokemon faints or if the player wins the game. Negative reward for a player pokemon fainting or losing/tying the game. PlayerAI then forwards the given action to Battle action queue.
4. When battle ends, either by a player running out of pokemon or time limit (240s) reached, Battle pushes final state-action-reward to Memory queue.
5. Battle calls PlayerModel to update Q-function with data in PlayerMemory queue and then train perceptron network on updated data.
6. ** Q-function and perceptron model are saved to json file **


What are the roles and responsibilities?:

TrainSetupInterface.js : get info from train setup webpage, pass along info to MatchHandler to start battle

MatchHandler.js : set up and start battles, begin post-battle analysis when battle ends

Battle.js : manage and update battle state at each turn, query actions from players, execute actions on action queue

Player.js : maintain a battling player state (num shields, pokemon team, switch cooldown)

TrainingAI.js/PlayerAI.js : choose a team, choose a battle action, choose shield, choose switch, guess opponent state (team,moves,stats)

Pokemon.js : maintain a battling pokemon state
