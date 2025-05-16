import numpy as np
from agent import Agents as BaseAgent
from collections import deque

class Edge:
    def __init__(self, to, rev, cap, cost):
        self.to = to
        self.rev = rev
        self.cap = cap
        self.cost = cost

class MCMFAgents(BaseAgent):
    def __init__(self):
        super().__init__()
        self.map = None
        self.n_rows = self.n_cols = 0
        self.robots = []
        self.packages = []
        self.free = []
        self.targets = []

    def init_agents(self, state):
        super().init_agents(state)
        self.map = state['map']
        self.n_rows = len(self.map)
        self.n_cols = len(self.map[0])
        self.robots = [(r[0]-1, r[1]-1, r[2]) for r in state['robots']]
        self.targets = [None] * len(self.robots)
        self.packages = []
        self.free = []
        for p in state['packages']:
            self.packages.append((p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5], p[6]))
            self.free.append(True)

    def valid(self, pos):
        r, c = pos
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols and self.map[r][c] == 0
    
    def bfs(self, start, goal):
        q = deque([(start, [])])
        seen = {start}
        while q:
            (x, y), path = q.popleft()
            if (x, y) == goal:
                return path
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nb = (x+dx, y+dy)
                if self.valid(nb) and nb not in seen:
                    seen.add(nb)
                    q.append((nb, path + [nb]))
        return []

    def min_cost_flow(self, N, S, T, maxf):
        res_flow, res_cost = 0, 0
        dist = [0]*N; prevv=[0]*N; preve=[0]*N
        while res_flow < maxf:
            INF = 10**9
            dist = [INF]*N
            inq = [False]*N
            dist[S] = 0; que=[S]; inq[S]=True
            for u in que:
                inq[u] = False
                for i, e in enumerate(self.graph[u]):
                    if e.cap>0 and dist[e.to]>dist[u]+e.cost:
                        dist[e.to]=dist[u]+e.cost
                        prevv[e.to]=u; preve[e.to]=i
                        if not inq[e.to]: que.append(e.to); inq[e.to]=True
            if dist[T]==INF: break
            d = maxf-res_flow; v=T
            while v!=S:
                d = min(d, self.graph[prevv[v]][preve[v]].cap)
                v = prevv[v]
            res_flow += d; res_cost += d*dist[T]
            v=T
            while v!=S:
                e=self.graph[prevv[v]][preve[v]]
                e.cap -= d
                self.graph[v][e.rev].cap += d
                v=prevv[v]
        return res_flow, res_cost

    def get_actions(self, state):
        t = state['time_step']
        self.robots = [(r[0]-1, r[1]-1, r[2]) for r in state['robots']]
        for p in state['packages']:
            tup = (p[0], p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5], p[6])
            if tup not in self.packages:
                self.packages.append(tup)
                self.free.append(True)
        n = len(self.robots)
        unassigned = [i for i,(rx,ry,car) in enumerate(self.robots) if car==0 and self.targets[i] is None]
        m = len(self.packages)
        if unassigned:
            Nnode=n+m+2; S=n+m; T=n+m+1
            self.graph=[[] for _ in range(Nnode)]
            def add_edge(u,v,cap,cost):
                self.graph[u].append(Edge(v,len(self.graph[v]),cap,cost))
                self.graph[v].append(Edge(u,len(self.graph[u])-1,0,-cost))
            for i in unassigned:
                for e in self.graph[i]:
                    if n<=e.to< n+m and e.cap==0:
                        j=e.to-n
                        self.targets[i]=j
                        self.free[j]=False
            for i, (rx, ry, car) in enumerate(self.robots):
                if car == 0 and self.targets[i] is None:
                    best_j = None; best_dist = float('inf')
                    for j, (pid, sx, sy, tx, ty, st, dl) in enumerate(self.packages):
                        if self.free[j] and t >= st:
                            path = self.bfs((rx, ry), (sx, sy))
                            if path and len(path) < best_dist:
                                best_j = j; best_dist = len(path)
                    if best_j is not None:
                        self.targets[i] = best_j
                        self.free[best_j] = False
        actions=[]
        for i,(rx,ry,car) in enumerate(self.robots):
            if car!=0:
                pkg=next(p for p in self.packages if p[0]==car)
                target=(pkg[3],pkg[4])
                act='2'
            elif self.targets[i] is not None:
                j=self.targets[i]; pkg=self.packages[j]
                target=(pkg[1],pkg[2])
                act='1'
            else:
                actions.append(('S','0'))
                continue
            path=self.bfs((rx,ry),target)
            if path:
                dx,dy=path[0][0]-rx,path[0][1]-ry
                move={(-1,0):'U',(1,0):'D',(0,-1):'L',(0,1):'R'}.get((dx,dy),'S')
            else:
                move='S'
            actions.append((move,act))
            if act=='1' and (rx,ry)==target:
                self.targets[i]=None
        return actions
