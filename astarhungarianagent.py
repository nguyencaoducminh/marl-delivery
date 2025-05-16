from scipy.optimize import linear_sum_assignment
import heapq

class AStarHungarianAgents:
    def __init__(self):
        self.is_init = False
        self.map = None
        self.robots = []
        self.robots_target = []
        self.packages = []
        self.packages_free = []
        self.n_robots = 0

    def init_agents(self, state):
        self.map = state['map']
        self.robots = [(r[0] - 1, r[1] - 1, 0) for r in state['robots']]
        self.n_robots = len(self.robots)
        self.robots_target = ['free'] * self.n_robots
        self.packages = [(p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5]) for p in state['packages']]
        self.packages_free = [True] * len(self.packages)

    def update_inner_state(self, state):
        for i, robot in enumerate(state['robots']):
            prev = self.robots[i]
            self.robots[i] = (robot[0]-1, robot[1]-1, robot[2])
            if prev[2] != 0 and self.robots[i][2] == 0:
                self.robots_target[i] = 'free'
            elif self.robots[i][2] != 0:
                self.robots_target[i] = self.robots[i][2]
        self.packages += [(p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5]) for p in state['packages']]
        self.packages_free += [True] * len(state['packages'])

    def a_star(self, start, goal):
        h = lambda x: abs(x[0] - goal[0]) + abs(x[1] - goal[1])
        open_set = [(h(start), 0, start)]
        # came_from = {}
        g_score = {start: 0}
        visited = set()

        while open_set:
            _, g, current = heapq.heappop(open_set)
            if current == goal:
                return g  # Only need cost
            if current in visited:
                continue
            visited.add(current)
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = current[0] + dx, current[1] + dy
                if 0 <= nx < len(self.map) and 0 <= ny < len(self.map[0]) and self.map[nx][ny] == 0:
                    neighbor = (nx, ny)
                    tentative_g = g + 1
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        heapq.heappush(open_set, (tentative_g + h(neighbor), tentative_g, neighbor))
        return 1e6  # Unreachable

    def get_move_direction(self, start, goal):
        min_cost = 1e6
        best_move = 'S'
        for move, (dx, dy) in zip(['U','D','L','R'], [(-1,0),(1,0),(0,-1),(0,1)]):
            nx, ny = start[0] + dx, start[1] + dy
            if 0 <= nx < len(self.map) and 0 <= ny < len(self.map[0]) and self.map[nx][ny] == 0:
                cost = self.a_star((nx, ny), goal)
                if cost < min_cost:
                    min_cost = cost
                    best_move = move
        return best_move

    def get_actions(self, state):
        if not self.is_init:
            self.is_init = True
            self.update_inner_state(state)
        else:
            self.update_inner_state(state)

        actions = []
        robot_positions = [(r[0], r[1]) for r in self.robots]
        available_packages = [i for i, free in enumerate(self.packages_free) if free]

        # Assign robots to packages with Hungarian matching (only if they are free)
        cost_matrix = []
        idle_robots = [i for i, t in enumerate(self.robots_target) if t == 'free']

        for i in idle_robots:
            row = []
            for j in available_packages:
                start_pos = robot_positions[i]
                package_pos = (self.packages[j][1], self.packages[j][2])
                delivery_pos = (self.packages[j][3], self.packages[j][4])

                # delivery_time = self.a_star(start_pos, package_pos) + self.a_star(package_pos, delivery_pos)
                pickup_time = self.a_star(start_pos, package_pos)
                delivery_time = self.a_star(package_pos, delivery_pos)
                # time_left = self.packages[j][5] - state['time_step']
                # lateness = delivery_time - time_left
                lateness = max(0, state['time_step'] + pickup_time + delivery_time - self.packages[j][5])

                cost = (pickup_time + 2 * delivery_time) + 0.3 * (lateness)

                row.append(cost)
            cost_matrix.append(row)

        if cost_matrix:
            # Linear sum assignment to assign which robot picks up which package
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                robot_idx = idle_robots[r]
                pkg_idx = available_packages[c]
                self.robots_target[robot_idx] = self.packages[pkg_idx][0]
                self.packages_free[pkg_idx] = False

        for i in range(self.n_robots):
            target_id = self.robots_target[i]
            if target_id == 'free':
                actions.append(('S', '0'))
                continue

            pkg_idx = target_id - 1
            if self.robots[i][2] != 0:
                # Carrying package: go to delivery location
                goal = (self.packages[pkg_idx][3], self.packages[pkg_idx][4])
                if (self.robots[i][0], self.robots[i][1]) == goal:
                    actions.append(('S', '2'))  # Drop
                else:
                    move = self.get_move_direction((self.robots[i][0], self.robots[i][1]), goal)
                    actions.append((move, '0'))
            else:
                # Go to pickup
                start = (self.robots[i][0], self.robots[i][1])
                goal = (self.packages[pkg_idx][1], self.packages[pkg_idx][2])
                if start == goal:
                    actions.append(('S', '1'))  # Pickup
                else:
                    move = self.get_move_direction(start, goal)
                    actions.append((move, '0'))
        return actions
