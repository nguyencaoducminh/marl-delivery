import numpy as np
from agent import Agents as BaseAgent
from collections import deque


class Edge:
    def __init__(self, to, rev, cap, cost):
        self.to = to          # đỉnh đích
        self.rev = rev        # vị trí cạnh ngược trong list của đỉnh to
        self.cap = cap        # capacity còn lại
        self.cost = cost      # chi phí


class MCMFAgents(BaseAgent):
    # ---------- khởi tạo & tiện ích -------------------------------------------------
    def __init__(self):
        super().__init__()
        self.map = None
        self.n_rows = self.n_cols = 0
        self.robots = []          # (row, col, carrying_id)
        self.packages = []        # (id, sx, sy, tx, ty, start, deadline)
        self.free = []            # True nếu gói chưa gán
        self.targets = []         # robot i -> chỉ số gói (hoặc None)

    def init_agents(self, state):
        super().init_agents(state)
        self.map = state['map']
        self.n_rows = len(self.map)
        self.n_cols = len(self.map[0])
        self.robots  = [(r[0]-1, r[1]-1, r[2]) for r in state['robots']]
        self.targets = [None] * len(self.robots)
        self.packages, self.free = [], []
        for p in state['packages']:
            self.packages.append((p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5], p[6]))
            self.free.append(True)

    # ---------- BFS ngắn nhất -------------------------------------------------------
    def valid(self, pos):
        r, c = pos
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols and self.map[r][c] == 0

    def bfs(self, start, goal):
        q = deque([(start, [])]); seen = {start}
        while q:
            (x, y), path = q.popleft()
            if (x, y) == goal:
                return path
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nb = (x+dx, y+dy)
                if self.valid(nb) and nb not in seen:
                    seen.add(nb)
                    q.append((nb, path+[nb]))
        return []

    # ---------- SSP + SPFA ----------------------------------------------------------
    def min_cost_flow(self, N, S, T, maxf):
        res_flow = res_cost = 0
        dist = [0]*N; prevv = [0]*N; preve = [0]*N
        while res_flow < maxf:
            INF = 10**9
            dist[:] = [INF]*N
            inq = [False]*N
            dist[S] = 0
            que = [S]; inq[S] = True
            for u in que:                # SPFA
                inq[u] = False
                for i, e in enumerate(self.graph[u]):
                    if e.cap and dist[e.to] > dist[u] + e.cost:
                        dist[e.to] = dist[u] + e.cost
                        prevv[e.to] = u
                        preve[e.to] = i
                        if not inq[e.to]:
                            que.append(e.to); inq[e.to] = True
            if dist[T] == INF: break     # hết đường tăng
            d = maxf - res_flow          # bottleneck
            v = T
            while v != S:
                d = min(d, self.graph[prevv[v]][preve[v]].cap)
                v = prevv[v]
            res_flow += d
            res_cost += d * dist[T]
            v = T
            while v != S:
                e = self.graph[prevv[v]][preve[v]]
                e.cap -= d
                self.graph[v][e.rev].cap += d
                v = prevv[v]
        return res_flow, res_cost

    # ---------- hàm chính mỗi bước --------------------------------------------------
    def get_actions(self, state):
        t = state['time_step']

        # ----------- cập nhật robots & packages -----------
        self.robots = [(r[0]-1, r[1]-1, r[2]) for r in state['robots']]
        for p in state['packages']:
            tup = (p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5], p[6])
            if tup not in self.packages:
                self.packages.append(tup); self.free.append(True)

        n, m = len(self.robots), len(self.packages)
        unassigned = [i for i,(x,y,car) in enumerate(self.robots)
                    if car == 0 and self.targets[i] is None]

        # ------------------- MCMF -------------------------
        if unassigned:
            Nnode = n + m + 2; S = n + m; T = n + m + 1
            self.graph = [[] for _ in range(Nnode)]

            def add_edge(u,v,cap,cost):
                self.graph[u].append(Edge(v,len(self.graph[v]),cap,cost))
                self.graph[v].append(Edge(u,len(self.graph[u])-1,0,-cost))

            # S → robot
            for i in unassigned:
                add_edge(S, i, 1, 0)

            # robot → package (gói còn free & đã xuất hiện)
            for i in unassigned:
                rx, ry, _ = self.robots[i]
                for j, (pid, sx, sy, tx, ty, st, dl) in enumerate(self.packages):
                    if self.free[j] and t >= st:
                        path = self.bfs((rx, ry), (sx, sy))
                        if path:
                            add_edge(i, n + j, 1, len(path))

            # package → T
            for j in range(m):
                if self.free[j]:
                    add_edge(n + j, T, 1, 0)

            # chạy SSP
            self.min_cost_flow(Nnode, S, T, len(unassigned))

            # đọc kết quả flow
            for i in unassigned:
                for e in self.graph[i]:
                    if n <= e.to < n + m and e.cap == 0:
                        j = e.to - n
                        self.targets[i] = j
                        self.free[j] = False
                        break
            # robot nào không được nối cạnh khả thi sẽ giữ targets[i] = None

        # ------------------ sinh hành động ----------------
        actions = []
        for i, (rx, ry, car) in enumerate(self.robots):
            if car:                              # đang mang hàng
                pkg = next(p for p in self.packages if p[0] == car)
                dest = (pkg[3], pkg[4]); pkg_act = '2'
            elif self.targets[i] is not None:    # đang đi lấy hàng
                pkg = self.packages[self.targets[i]]
                dest = (pkg[1], pkg[2]); pkg_act = '1'
            else:                                # rảnh, nhưng không có gói khả thi
                actions.append(('S', '0')); continue

            path = self.bfs((rx, ry), dest)
            move = 'S'
            if path:
                dx, dy = path[0][0] - rx, path[0][1] - ry
                move = {(-1,0): 'U', (1,0): 'D', (0,-1): 'L', (0,1): 'R'}.get((dx, dy), 'S')

            actions.append((move, pkg_act))

            # nếu vừa tới điểm lấy hàng, xoá target để vòng sau chuyển sang trạng thái 'đang mang'
            if pkg_act == '1' and (rx, ry) == dest:
                self.targets[i] = None

        return actions

