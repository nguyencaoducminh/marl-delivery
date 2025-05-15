# import numpy as np
# Run a BFS to find the path from start to goal
def run_bfs(map, start, goal):
    n_rows = len(map)
    n_cols = len(map[0])

    queue = []
    visited = set()
    queue.append((goal, []))
    visited.add(goal)
    d = {}
    d[goal] = 0

    while queue:
        current, path = queue.pop(0)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (current[0] + dx, current[1] + dy)
            if next_pos[0] < 0 or next_pos[0] >= n_rows or next_pos[1] < 0 or next_pos[1] >= n_cols:
                continue
            if next_pos not in visited and map[next_pos[0]][next_pos[1]] == 0:
                visited.add(next_pos)
                d[next_pos] = d[current] + 1
                queue.append((next_pos, path + [next_pos]))

    if start not in d:
        return 'S', 100000
    
    t = 0
    actions = ['U', 'D', 'L', 'R']
    current = start
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        next_pos = (current[0] + dx, current[1] + dy)
        if next_pos in d:
            if d[next_pos] == d[current] - 1:
                return actions[t], d[next_pos]
        t += 1
    return 'S', d[start]

class Graph:

    def __init__(self, distances):
        self.edges = []
        self.n_robots = len(distances)
        self.n_packages = len(distances[0])

        for i in range(len(distances)):
            for package_id, distance in distances[i].items():
                self.edges.append((i, package_id, distance))
    
    def print(self):
        for edge in self.edges:
            print(edge)

    def match_robot_package(self):
        self.edges.sort(key=lambda x: x[2])

        matches = {}

        for edge in self.edges:
            if edge[0] not in matches.keys():
                if edge[1] not in matches.values():
                    matches[edge[0]] = edge[1]
        
        return matches

class Agents:

    def __init__(self):
        self.agents = []
        self.packages = []
        self.packages_free = []
        self.n_robots = 0
        self.state = None

        self.is_init = False

    def init_agents(self, state):
        self.state = state
        self.n_robots = len(state['robots'])
        self.map = state['map']
        self.robots = [(robot[0]-1, robot[1]-1, 0) for robot in state['robots']]
        self.robots_target = ['free'] * self.n_robots
        self.packages += [(p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5]) for p in state['packages']]

        self.packages_free = [True] * len(self.packages)

    def update_move_to_target(self, robot_id, target_package_id, phase='start'):

        if phase == 'start':
            distance = abs(self.packages[target_package_id][1]-self.robots[robot_id][0]) + \
            abs(self.packages[target_package_id][2]-self.robots[robot_id][1])
        else:
            # Switch to the distance to target (3, 4) if phase == 'target'
            distance = abs(self.packages[target_package_id][3]-self.robots[robot_id][0]) + \
            abs(self.packages[target_package_id][4]-self.robots[robot_id][1])
        i = robot_id
        #print(self.robots[i], distance, phase)

        # Step 4: Move to the package
        pkg_act = 0
        move = 'S'
        if distance >= 1:
            pkg = self.packages[target_package_id]
            
            target_p = (pkg[1], pkg[2])
            if phase == 'target':
                target_p = (pkg[3], pkg[4])
            move, distance = run_bfs(self.map, (self.robots[i][0], self.robots[i][1]), target_p)

            if distance == 0:
                if phase == 'start':
                    pkg_act = 1 # Pickup
                else:
                    pkg_act = 2 # Drop
        else:
            move = 'S'
            pkg_act = 1    
            if phase == 'start':
                pkg_act = 1 # Pickup
            else:
                pkg_act = 2 # Drop    

        return move, str(pkg_act)
    
    def update_inner_state(self, state):
        # Update robot positions and states
        for i in range(len(state['robots'])):
            prev = (self.robots[i][0], self.robots[i][1], self.robots[i][2])
            robot = state['robots'][i]
            self.robots[i] = (robot[0]-1, robot[1]-1, robot[2])
            # print(i, self.robots[i])
            if prev[2] != 0:
                if self.robots[i][2] == 0:
                    # Robot has dropped the package
                    self.robots_target[i] = 'free'
                else:
                    self.robots_target[i] = self.robots[i][2]
        
        # Update package positions and states
        self.packages += [(p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5]) for p in state['packages']]
        self.packages_free += [True] * len(state['packages'])    

    def get_distance_to_all_packages(self, robot):
        distances = {}
        robot_pos = (robot[0], robot[1])
        for i in range(len(self.packages)):
            if not self.packages_free[i]:
                continue
            package_id = self.packages[i][0]
            package_pos = (self.packages[i][1], self.packages[i][2])
            move, distance = run_bfs(self.map, robot_pos, package_pos)
            if distance != 100000:
                distances[package_id] = distance

        return distances

    def find_collision(self, positions):
        positions_dict = {}
        conflicts = []

        for i, pos in enumerate(positions):
            if pos in positions_dict:
                conflicts.append((positions_dict[pos], i))
            else:
                positions_dict[pos] = i

        return conflicts

    def resolve_collision(self, actions):
        current_pos = []
        new_pos = []
        new_actions = actions

        for i in range(self.n_robots):
            current_pos.append((self.robots[i][0], self.robots[i][1]))
            
            if actions[i][0] == 'U':
                new_pos.append((self.robots[i][0], self.robots[i][1]-1))
            elif actions[i][0] == 'D':
                new_pos.append((self.robots[i][0], self.robots[i][1]+1))
            elif actions[i][0] == 'L':
                new_pos.append((self.robots[i][0]-1, self.robots[i][1]))
            elif actions[i][0] == 'R':
                new_pos.append((self.robots[i][0]+1, self.robots[i][1]))
            elif actions[i][0] == 'S':
                new_pos.append((self.robots[i][0], self.robots[i][1]))
        
        conflicts = self.find_collision(new_pos)

        # If there are no conflicts, return original actions
        if len(conflicts) < 0:
            return new_actions

        for conflict in conflicts:
            print(conflict)
            robot1, robot2 = conflict

            movement_offsets = {
                'U': (0, -1),
                'D': (0, 1),
                'L': (-1, 0),
                'R': (1, 0)
            }
            
            if actions[robot1][0] == 'S' or actions[robot2][0] == 'S':
                if actions[robot1][1] != '0':
                    new_actions[robot2] = ('S', '0')
                elif actions[robot2][1] != '0':
                    new_actions[robot1] = ('S', '0')
                else:
                    if actions[robot1][0] == 'S':
                        piority = robot2
                        other = robot1
                    else:
                        piority = robot1
                        other = robot2

                    if actions[piority][0] == 'U':
                        posible_actions = ['L', 'R', 'U']
                    elif actions[piority][0] == 'D':
                        posible_actions = ['L', 'R', 'D']
                    elif actions[piority][0] == 'L':
                        posible_actions = ['U', 'D', 'L']
                    elif actions[piority][0] == 'R':
                        posible_actions = ['U', 'D', 'R']

                    current_x, current_y = self.robots[other][0], self.robots[other][1]

                    for action in posible_actions:
                        if action in movement_offsets:
                            dx, dy = movement_offsets[action]
                            new_location = (current_x + dx, current_y + dy)

                            if new_location not in new_pos and self.map[new_location[0]][new_location[1]] == 0:
                                new_actions[robot1] = (action, '0')
                                break

            else:
                pack1 = self.packages[self.robots_target[robot1]-1] 
                pack2 = self.packages[self.robots_target[robot2]-1]

                des1 = (pack1[1], pack1[2]) if self.robots[robot1][2] == 0 else (pack1[3], pack1[4])
                des2 = (pack2[1], pack2[2]) if self.robots[robot2][2] == 0 else (pack2[3], pack2[4])

                m1,dist1 = run_bfs(self.map, (self.robots[robot1][0], self.robots[robot1][1]), des1)
                m2,dist2 = run_bfs(self.map, (self.robots[robot2][0], self.robots[robot2][1]), des2)

                if dist1 >= dist2:
                    piority = robot1
                    other = robot2
                else:
                    piority = robot2
                    other = robot1
                
                if actions[piority][0] == 'U':
                    posible_actions = ['L', 'R', 'U']
                elif actions[piority][0] == 'D':
                    posible_actions = ['L', 'R', 'D']
                elif actions[piority][0] == 'L':
                    posible_actions = ['U', 'D', 'L']
                elif actions[piority][0] == 'R':
                    posible_actions = ['U', 'D', 'R']

                current_x, current_y = self.robots[other][0], self.robots[other][1]

                for action in posible_actions:
                    if action in movement_offsets:
                        dx, dy = movement_offsets[action]
                        new_location = (current_x + dx, current_y + dy)

                        if new_location not in new_pos and self.map[new_location[0]][new_location[1]] == 0:
                            new_actions[robot1] = (action, '0')
                            break
                        
        return new_actions

    def get_actions(self, state):
        if self.is_init == False:
            # This mean we have invoke the init agents, use the update_inner_state to update the state
            self.is_init = True
            self.update_inner_state(state)

            # distances = [self.get_distance_to_all_packages(self.robots[i]) for i in range(self.n_robots)]

            # graph = Graph(distances)
            # matches = graph.match_robot_package()
            
            # for robot_id, package_id in matches.items():
            #     self.robots_target[robot_id] = package_id
            #     self.packages_free[package_id-1] = False
            #     # print("Robot", robot_id, "assigned to package", package_id)
            #     self.update_move_to_target(robot_id, package_id-1)

        else:
            self.update_inner_state(state)

        actions = []
        print("State robot: ", self.robots)

        free_robots = [i for i in range(self.n_robots) if self.robots_target[i] == 'free']
        if len(free_robots) >= 2:
            distances = [self.get_distance_to_all_packages(self.robots[i]) for i in free_robots]
            graph = Graph(distances)
            matches = graph.match_robot_package()
            for robot_id, package_id in matches.items():
                self.robots_target[robot_id] = package_id
                self.packages_free[package_id-1] = False
                # print("Robot", robot_id, "assigned to package", package_id)
                self.update_move_to_target(robot_id, package_id-1)

        # Start assigning a greedy strategy
        for i in range(self.n_robots):
            # Step 1: Check if the robot is already assigned to a package
            if self.robots_target[i] != 'free':
                
                current_package = self.robots_target[i]
                # Step 1b: Check if the robot has reached the package
                if self.robots[i][2] != 0:
                    # Move to the target points
                    
                    move, action = self.update_move_to_target(i, current_package-1, 'target')
                    actions.append((move, str(action)))
                else:  
                    # Step 1c: Continue to move to the package
                    move, action = self.update_move_to_target(i, current_package-1)    
                    actions.append((move, str(action)))
            else:
                # Step 2: Find a package to pick up
                # Find the closest package
                closest_package_id = None
                closed_distance = 1000000
                for j in range(len(self.packages)):
                    if not self.packages_free[j]:
                        continue

                    pkg = self.packages[j]                
                    m, d = run_bfs(self.map, (pkg[1], pkg[2]), (self.robots[i][0], self.robots[i][1]))
                    if d < closed_distance:
                        closed_distance = d
                        closest_package_id = pkg[0]

                if closest_package_id is not None:
                    self.packages_free[closest_package_id-1] = False
                    self.robots_target[i] = closest_package_id
                    move, action = self.update_move_to_target(i, closest_package_id-1)    
                    actions.append((move, str(action)))
                else:
                    actions.append(('S', '0'))

        actions = self.resolve_collision(actions)

        print("N robots = ", len(self.robots))
        print("Actions = ", actions)
        print(self.robots_target)
        return actions
