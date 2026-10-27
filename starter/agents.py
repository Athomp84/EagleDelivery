from collections import deque
import heapq
from itertools import count


class SearchResult:
    def __init__(self, path, expanded):
        self.path = path
        self.expanded = expanded


def reconstruct_path(parents, goal):
    path = []
    state = goal
    while state is not None:
        path.append(state)
        state = parents[state]
    path.reverse()
    return path


def bfs(campus, start, goal):
    frontier = deque([start])
    parents = {start: None}
    expanded = 0
    while frontier:
        state = frontier.popleft()
        if state == goal:
            return SearchResult(reconstruct_path(parents, goal), expanded)
        expanded += 1
        for action, successor in campus.neighbors(state):
            if successor not in parents:
                parents[successor] = state
                frontier.append(successor)
    return SearchResult(None, expanded)


def manhattan(state, goal):
    """TODO 1: Return Manhattan distance between two (row, column) tuples."""
    # Add the absolute row difference and the absolute column difference.
    # raise NotImplementedError("Complete TODO 1: manhattan")
    row, col = state
    goal_row, goal_col = goal
    
    return abs(row-goal_row) + abs(col-goal_col)


def astar(campus, start, goal):
    """TODO 2: Return SearchResult(path, expanded) for A* graph search."""
    # Use heapq as a priority queue and manhattan() as the heuristic.
    # Track best known g-costs and parent states in dictionaries.

    # catch the starting edge case before doing any work.
    if start == goal:
        return SearchResult([start],0)

    best_g = {start : 0}
    parents = {start : None}
    serial = count()
    frontier = [(manhattan(start, goal), next(serial), 0, start)]
    expanded = 0

    # heavily lifted from class example
    while frontier:
        # pull from heap
        f_cost,_,g_cost, state = heapq.heappop(frontier)
        
        # if better path to current state already expanded skip this expansion 
        if g_cost > best_g.get(state, float("inf")):
            continue

        # goal check, return path and results
        if state == goal:
            return SearchResult(reconstruct_path(parents, goal), expanded)

        # count expansion
        expanded += 1

        # get actions and next states relative to current state
        for action, next_state in campus.neighbors(state):
            new_g = g_cost + 1
            
            # if a better path has been found update best_g, parents and push to frontier
            if new_g < best_g.get(next_state, float("inf")):
                best_g[next_state] = new_g
                parents[next_state] = state
                new_f = new_g + manhattan(next_state, goal)
                heapq.heappush(frontier, (new_f, next(serial), new_g, next_state))
    return SearchResult(None, expanded)

def epsilon_greedy(q_values, state, actions, epsilon, rng):
    """TODO 3: Choose an action with epsilon-greedy exploration."""
    # q_values is a dictionary. Its keys are (state, action) pairs.
    # Example key: ((1, 1), "RIGHT"). Unseen entries have value 0.0.
    #raise NotImplementedError("Complete TODO 3: epsilon_greedy")

    # choose uniformly among all 4 actions, probability epsilon.
    if rng.random() < epsilon:
        return rng.choice(actions)

    best_actions = [] 
    max_q = float("-inf")

    # track the best q_val and its relative action
    for action in actions:
        q_val = q_values.get((state, action), 0.0)

        if q_val > max_q:
            max_q = q_val
            best_actions = [action]
        elif q_val == max_q:
            best_actions.append(action)

    # choose randomly from actions with the best q_vals, breaks ties uniformly.
    return rng.choice(best_actions)

def q_learning_update(q_values, state, action, reward, next_state,
                      actions, alpha, gamma, terminal):
    """TODO 4: Update exactly one Q-table entry in place; return nothing."""
    # Get Q(state, action), defaulting to 0.0.
    # If terminal is True, the target is reward with no continuation term.
    # Otherwise target = reward + gamma * max Q(next_state, next_action).
    # Consider every action, including unseen entries with value 0.0.
    # Apply Q <- Q + alpha * (target - Q).
    #raise NotImplementedError("Complete TODO 4: q_learning_update")

    current_q_val = q_values.get((state, action), 0.0)

    if terminal:
        target = reward
    else:
        max_next_q_val = max(q_values.get((next_state,next_action),0.0) for next_action in actions)
        target = reward + gamma * max_next_q_val

    q_values[(state, action)] = current_q_val + alpha * (target - current_q_val) 