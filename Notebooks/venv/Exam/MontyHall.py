import sys
import numpy as np
import matplotlib.pyplot as plt

def control(choice, prize):
    if choice == prize:
        return 1  
    else:
        return 0  

def game(N, M=2, switch=False):
    """
    Simulates the Monty Hall game.
    Returns the index of the box containing the prize.
    """
    boxes = np.zeros(N)  # Array of N empty boxes
    prize_box_index = np.random.randint(0, N)  # Index if the box containing the prize
    boxes[prize_box_index] = 1  # Mark the prize box with 1
    contestant_choice = np.random.randint(0, N)  # Contestant's initial choice
    
    opened_boxes = np.random.choice(
    [i for i in range(N) if i != contestant_choice and i != prize_box_index],
    size=N-M, replace=False
    ) # the opened boxes should not contain the prize and the contestant's choice
    
    if (contestant_choice in opened_boxes) or (prize_box_index in opened_boxes):
        raise ValueError("Contestant's choice or prize box cannot be in opened boxes.") #just a check...
    
    third_gamer_choice =  np.random.choice(
        [i for i in range(N) if i not in opened_boxes],
        size=1, replace=False
    )[0] # third gamer doesn't know what happend before, he just sees N-M boxes to randomly choose from
    
    first_gamer = control(contestant_choice, prize_box_index) 
    remaining_boxes = [i for i in range(N) if i != contestant_choice and i not in opened_boxes]
    new_choice = np.random.choice(remaining_boxes)  # Contestant switches to one of the remaining boxes
    second_gamer = control(new_choice, prize_box_index)
    third_gamer = control(third_gamer_choice, prize_box_index) 

    return first_gamer, second_gamer, third_gamer

def run_simulation(N, M=2, trials=100):
    """
    Runs the Monty Hall simulation for a given number of trials.
    N: Number of boxes
    trials: Number of game trials
    switch: Whether the contestant switches their choice
    Returns the win rate.
    """
    fg_wins = []
    sg_wins = []
    tg_wins = []
    for _ in range(trials):
        result = game(N, M, False)
        fg_wins.append(result[0])
        sg_wins.append(result[1])
        tg_wins.append(result[2])
    
    no_switch_win_rate = np.array(fg_wins).sum() / trials
    switch_win_rate    = np.array(sg_wins).sum() / trials
    tg_win_rate        = np.array(tg_wins).sum() / trials
    
    print(f'Win rate without switching: {no_switch_win_rate:.2f}')
    print(f'Win rate with switching: {switch_win_rate:.2f}')
    xplot = np.linspace(1,trials+1,trials)
    plt.figure(figsize=(14, 6))
    plt.plot(xplot, np.cumsum(fg_wins)/xplot, label='No Switch Wins', linewidth=2, color='blue')
    plt.plot(xplot, np.cumsum(sg_wins)/xplot, label=f'Switch Wins', linewidth=2, color='orange')
    plt.plot(xplot, np.cumsum(tg_wins)/xplot, label='Third Gamer Wins', linewidth=2, color='green')
    plt.axhline(y=no_switch_win_rate, color='blue', linestyle='--', label=f'No switch win rate : {np.mean(np.cumsum(fg_wins)/xplot):.2}')
    plt.axhline(y=switch_win_rate, color='orange', linestyle='--', label=f'Switch win rate : {np.mean(np.cumsum(sg_wins)/xplot):.2}')
    plt.axhline(y=tg_win_rate, color='green', linestyle='--', label=f'Third gamer win rate : {np.mean(np.cumsum(tg_wins)/xplot):.2}')
    plt.xlabel('Trial Number')
    plt.ylabel('Number of wins')
    plt.title(f'Monty Hall Simulation Results: N = {N}, Trials = {trials}')
    plt.legend()
    plt.grid()
    plt.show()    
    

N = int(sys.argv[1])  # Number of boxes
M = int(sys.argv[2])  # Number of boxes
trials = int(sys.argv[3])  # Number of trials
if trials < N:
    raise ValueError("Number of trials must be greater than or equal to the number of boxes.")
if M<2:
    raise ValueError("Number of remaining boxes must be greater than or equal to 2.")
run_simulation(N, M, trials)  # Run the simulation