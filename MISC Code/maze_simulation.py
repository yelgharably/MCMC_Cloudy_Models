import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.animation as animation

def create_maze(dim):
    # Create a grid filled with walls
    maze = np.ones((dim*2+1, dim*2+1))

    # Define the starting point
    x, y = (0, 0)
    maze[2*x+1, 2*y+1] = 0

    # Initialize the stack with the starting point
    stack = [(x, y)]
    while len(stack) > 0:
        x, y = stack[-1]

        # Define possible directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if nx >= 0 and ny >= 0 and nx < dim and ny < dim and maze[2*nx+1, 2*ny+1] == 1:
                maze[2*nx+1, 2*ny+1] = 0
                maze[2*x+1+dx, 2*y+1+dy] = 0
                stack.append((nx, ny))
                break
        else:
            stack.pop()
            
    # Create an entrance and an exit
    maze[1, 0] = 0
    maze[-2, -1] = 0

    return maze

def explore_maze(maze, start=(1, 1)):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    stack = [(start, [start])]
    visited = set()
    visited.add(start)
    junctions = []  # To store junctions
    all_paths = []
    current_path = []

    while stack:
        (x, y), path = stack.pop()
        current_path = path.copy()
        unvisited_neighbors = []

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0 and (nx, ny) not in visited:
                unvisited_neighbors.append((nx, ny))
                visited.add((nx, ny))

        if unvisited_neighbors:
            if len(unvisited_neighbors) > 1:
                junctions.append((x, y, path))
            for nx, ny in unvisited_neighbors:
                stack.append(((nx, ny), path + [(nx, ny)]))
        else:
            all_paths.append(path)
            if junctions:
                last_junction = junctions.pop()
                stack.append((last_junction[:2], last_junction[2]))

    return all_paths

def draw_maze(maze, paths=None, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set the border color to white
    fig.patch.set_edgecolor('white')
    fig.patch.set_linewidth(0)

    ax.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    
    if paths is not None:
        line, = ax.plot([], [], color='red', linewidth=2)
        frames = sum(len(path) for path in paths)

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            path_index, point_index = 0, frame
            for path in paths:
                if point_index < len(path):
                    break
                point_index -= len(path)
                path_index += 1
            
            path = paths[path_index]
            line.set_data(*zip(*[(p[1], p[0]) for p in path[:point_index+1]]))
            return line,

        max_path_length = max(len(path) for path in paths)
        ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, repeat=False, interval=1)  # Lower interval for faster movement

        if save_path:
            ani.save(save_path, writer='ffmpeg', fps=60)
    
    ax.arrow(0, 1, .4, 0, fc='green', ec='green', head_width=0.3, head_length=0.3)
    ax.arrow(maze.shape[1] - 1, maze.shape[0]  - 2, 0.4, 0, fc='blue', ec='blue', head_width=0.3, head_length=0.3)
    
    plt.show()    

if __name__ == "__main__":
    dim = int(input("Enter the dimension of the maze: "))
    maze = create_maze(dim)
    paths = explore_maze(maze)
    save_path = 'maze_exploration.mp4'
    draw_maze(maze, paths, save_path=save_path)
