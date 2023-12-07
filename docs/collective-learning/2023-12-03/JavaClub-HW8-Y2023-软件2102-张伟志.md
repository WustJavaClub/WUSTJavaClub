---
author : Cattle_Horse
create : 2023/12/5
---

# 第八次作业报告

姓名：张伟志

班级：软件2102

## 题目

### 1443. 收集树上所有苹果的最少时间

题目链接：[1443. 收集树上所有苹果的最少时间](https://leetcode.cn/problems/minimum-time-to-collect-all-apples-in-a-tree/description/)

时间复杂度：$O(n)$

空间复杂度：$O(n)$

由于需要重新回到0号节点，因此对于每一条需要走的边都会走两遍

深搜遍历并记录路径节点，判断子树是否有苹果，如果有苹果就将路径上的节点加入集合

最后所需要的边的个数为 `集合内节点个数 - 1`，而所需时间为 `边数 * 2`

```java
class Solution {
    ArrayList<ArrayList<Integer>> adj;
    List<Boolean> hashApple;
    HashSet<Integer> ans;
    
    /**
     * @return 返回以current为根节点的树是否有苹果
     */
    private boolean dfs(int current, boolean[] visited, ArrayList<Integer> path) {
        boolean childrenHasApple = false;
        for (Integer v : adj.get(current)) {
            if (!visited[v]) {
                visited[v] = true;
                path.add(v);
                childrenHasApple = dfs(v, visited, path) || childrenHasApple;
                path.remove(path.size() - 1);
            }
        }
        if (childrenHasApple) {
            return true;
        }
        if (hashApple.get(current)) {
            ans.addAll(path);
            return true;
        }
        return false;
    }

    int n, m;

    public int minTime(int n, int[][] edges, List<Boolean> hasApple) {
        this.n = n;
        this.m = edges.length;
        this.hashApple = hasApple;
        this.ans = new HashSet<>(n);
        this.adj = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            adj.add(new ArrayList<>());
        }
        for (int i = 0; i < m; ++i) {
            int from = edges[i][0], to = edges[i][1];
            adj.get(from).add(to);
            adj.get(to).add(from);
        }
        ArrayList<Integer> temp = new ArrayList<>(m);
        temp.add(0);
        ans.add(0);
        boolean[] visited = new boolean[n];
        visited[0] = true;
        dfs(0, visited, temp);
        return (ans.size() - 1) * 2;
    }
}
```

### 797. 所有可能的路径

题目链接：[797. 所有可能的路径](https://leetcode.cn/problems/all-paths-from-source-to-target/)

时间复杂度：$O(n\times 2^n)$

空间复杂度：$O(n)$

```java
class Solution {
    int n;
    int[][] adj;

    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        this.n = graph.length;
        this.adj = graph;
        List<List<Integer>> ans = new ArrayList<>();
        dfs(0, new ArrayList<>(n), ans);
        return ans;
    }

    private void dfs(int current, ArrayList<Integer> path, List<List<Integer>> ans) {
        if (current == n - 1) {
            ArrayList<Integer> temp = new ArrayList<>(path);
            temp.add(n - 1);
            ans.add(temp);
            return;
        }
        path.add(current);
        for (int v : adj[current]) {
            dfs(v, path, ans);
        }
        path.remove(path.size() - 1);
    }
}
```

### 207. 课程表

题目链接：[207. 课程表](https://leetcode.cn/problems/course-schedule/)

时间复杂度：$O(n+m)$，其中 $n$ 为课程个数，$m$ 为边数

空间复杂度：$O(n+m)$

参见[拓扑排序](###拓扑排序)

```java
public class Solution {
    public boolean canFinish(int n, int[][] edges) {
        List<Integer>[] adj = new List[n];
        Arrays.setAll(adj, value -> new ArrayList<>());
        for (int[] edge : edges) {
            adj[edge[1]].add(edge[0]);
        }
        return new DFSTopological().applyUnweighted(new int[n], adj);
    }
}
```

### 210. 课程表 II

题目链接：[210. 课程表 II](https://leetcode.cn/problems/course-schedule-ii/)

时间复杂度：$O(n+m)$，其中 $n$ 为课程个数，$m$ 为边数

空间复杂度：$O(n+m)$

同[207.课程表](##207.%20课程表)，参见[拓扑排序](###拓扑排序)

```java
public class Solution {
    public int[] findOrder(int n, int[][] edges) {
        List<Integer>[] adj = new List[n];
        Arrays.setAll(adj, value -> new ArrayList<>());
        for (int[] edge : edges) {
            adj[edge[1]].add(edge[0]);
        }
        int[] ans = new int[n];
        if (!new DFSTopological().applyUnweighted(ans, adj)) {
            ans = new int[0];
        }
        return ans;
    }
}
```

### 1584. 连接所有点的最小费用

题目链接：[1584. 连接所有点的最小费用](https://leetcode.cn/problems/min-cost-to-connect-all-points/)

时间复杂度：$O(n^2)$，不使用优先队列的Prim算法时间复杂度为 $O(n^2+m)$，其中 m 为 边数，$m=n\times (n-1)$

空间复杂度：$O(n\times n)$

在任意两个点间添加一条权重为曼哈顿距离的边，以此建图

这是一个完全图，稠密图

因此使用邻接矩阵存储图

使用Prim算法（且不使用优先队列）求[最小生成树](###最小生成树)

```java
class Solution {
    private int distance(int x1, int y1, int x2, int y2) {
        return Math.abs(x1 - x2) + Math.abs(y1 - y2);
    }

    public int minCostConnectPoints(int[][] points) {
        int n = points.length;
        int[][] graph = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    graph[i][j] = distance(points[i][0], points[i][1], points[j][0], points[j][1]);
                }
            }
        }
        return new Prim().apply(graph, 0x3f3f3f3f);
    }
}
```

### 1334. 阈值距离内邻居最少的城市

题目链接：[1334. 阈值距离内邻居最少的城市](https://leetcode.cn/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/description/)

时间复杂度：$O(n^3)$

空间复杂度：$O(n^2)$

节点到其余节点的最短路径长度小于等于 `distanceThreshold` 的个数，找到这个个数的最小值

使用 Floyd 求全源[最短路](###最短路)

```java
class Solution {
    public int findTheCity(int n, int[][] edges, int distanceThreshold) {
        int[][] distance = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                distance[i][j] = i == j ? 0 : 0x3f3f3f3f;
            }
        }
        for (int[] edge : edges) {
            int from = edge[0], to = edge[1], weight = edge[2];
            distance[from][to] = distance[to][from] = weight;
        }
        new Floyd().floyd(distance, n);
        int number = -1, count = n;
        for (int i = 0; i < n; i++) {
            // 求i的邻居城市有多少个
            int current = (int) Arrays.stream(distance[i]).filter(value -> value <= distanceThreshold).count() - 1;
            if (current <= count) {
                number = i;
                count = current;
            }
        }
        return number;
    }
}
```

### 2049. 统计最高分的节点数目

题目链接：[2049. 统计最高分的节点数目](https://leetcode.cn/problems/count-nodes-with-the-highest-score/description/)

如果对每一个节点都考虑删去后遍历计算子树的节点数，意味着对于任意节点都需要 $O(n)$ 的时间复杂度计算得分，总时间复杂度为 $O(n^2)$，会超时

```java
// 超时 O( n^2 )
public class Problem2049 {
    private List<Integer>[] adj;

    public int countHighestScoreNodes(int[] parents) {
        int n = parents.length;
        adj = new List[n];
        Arrays.setAll(adj, value -> new ArrayList<>());
        for (int i = 1; i < n; i++) {
            adj[parents[i]].add(i);
            adj[i].add(parents[i]);
        }
        int maxScore = 0, maxCount = 0;
        for (int i = 0; i < n; i++) {
            int score = 1;
            for (Integer v : adj[i]) {
                score *= calculateSize(v, i);
            }
            if (score > maxScore) {
                maxScore = score;
                maxCount = 1;
            } else if (score == maxScore) {
                ++maxCount;
            }
        }
        return maxCount;
    }

    /**
     * dfs计算子树节点个数
     */
    private int calculateSize(int current, int parent) {
        int size = 1;
        for (Integer v : adj[current]) {
            if (v != parent) {
                size += calculateSize(v, current);
            }
        }
        return size;
    }
}
```

#### 方法一

时间复杂度：$O(n)$

空间复杂度：$O(n)$

每一个节点的得分可以分为子节点 和 父节点 两部分，即
$$
\begin{aligned}
score &= size_{left}\times size_{right}\times size_{father}\\
&= size_{left}\times size_{right}\times(n-size_{left}-size_{right}-1)
\end{aligned}
$$
其中，$score$ 为节点 $x$ 的得分，$size_{left}$ 为节点 $x$ 的左子树节点个数，$size_{right}$ 同理。

$size$ 可以通过 深度优先搜索求得

注意：

1. 特判根节点
2. 使用 $long$ 存储得分

```java
class Solution {
    private int n;
    private List<Integer>[] adj;
    private long[] scores;

    public int countHighestScoreNodes(int[] parents) {
        n = parents.length;
        adj = new List[n];
        Arrays.setAll(adj, value -> new ArrayList<>());
        for (int i = 1; i < n; i++) {
            adj[parents[i]].add(i);
            adj[i].add(parents[i]);
        }
        scores = new long[n];
        Arrays.fill(scores, 1);
        calculateScores(0, -1);
        // 计算最高得分的个数
        long maxScore = -1;
        int maxCount = -1;
        for (long score : scores) {
            if (score > maxScore) {
                maxScore = score;
                maxCount = 1;
            } else if (score == maxScore) {
                ++maxCount;
            }
        }
        return maxCount;
    }

    /**
     * @return 以current为根节点的树的节点个数
     */
    private int calculateScores(int current, int parent) {
        int size = 1;
        for (Integer v : adj[current]) {
            if (v != parent) {
                int res = calculateScores(v, current);
                size += res;
                scores[current] *= res;
            }
        }
        // 特判根节点
        if (n - size != 0) {
            scores[current] *= n - size;
        }
        return size;
    }
}
```

#### 方法二

时间复杂度：$O(n)$

空间复杂度：$O(n)$

可以发现方法一的思路类似于拓扑排序，一直选择入度为0的点，然后删去，找下一个入度为0的点

```java
class Solution {
    public int countHighestScoreNodes(int[] parents) {
        /**
         * n : 节点个数
         * inDegree : 入度
         * size : 子树节点个数
         * scores : 得分
         */
        int n = parents.length;
        int[] inDegree = new int[n];
        int[] size = new int[n];
        long[] scores = new long[n];
        Arrays.fill(size, 1);
        Arrays.fill(scores, 1);
        for (int i = 1; i < n; i++) {
            ++inDegree[parents[i]];
        }
        // 入度为0的节点编号
        Deque<Integer> deque = new ArrayDeque<>(n);
        for (int i = 0; i < n; ++i) {
            if (inDegree[i] == 0) {
                deque.addLast(i);
            }
        }
        // 逐渐删去入度为0的节点，同时为其父节点的增加得分
        while (!deque.isEmpty()) {
            int index = deque.pollFirst();
            int parent = parents[index];
            if (--inDegree[parent] == 0 && parent != 0) {
                deque.addLast(parent);
            }
            size[parent] += size[index];
            scores[parent] *= size[index];
            scores[index] *= n - size[index];
        }
        // 计算最高得分的个数
        long maxScore = -1;
        int maxCount = -1;
        for (long score : scores) {
            if (score > maxScore) {
                maxScore = score;
                maxCount = 1;
            } else if (score == maxScore) {
                ++maxCount;
            }
        }
        return maxCount;
    }
}
```

### 851. 喧闹和富有

题目链接：[851. 喧闹和富有](https://leetcode.cn/problems/loud-and-rich/description/)

时间复杂度：$O(n+m)$ ，其中 $n$ 为人数，$m$ 为 `richer` 数组长度

空间复杂度：$O(n)$

```
样例 1 :
Input :
    richer = [[1,0],[2,1],[3,1],[3,7],[4,3],[5,3],[6,3]]
    quiet = [3,2,5,4,6,1,7,0]
Output :
    ans = [5,5,2,5,4,5,6,7]
```

定义 $a$ 指向 $b$ 表示 $a$ 比 $b$ 有钱，则对于样例 $1$，有下图

![leetcode851](image/leetcode851.svg)

其中，黑色数字表示 `person` 的标号，红色数字表示对应标号的 `quiet` 值

则题目的结果数组 $ans_x$ 表示：终点为 $x$ 的链中红色数字最小的下标

每一个点的结果值依赖于指向它的节点，因此通过拓扑排序优先求解依赖项

```java
class Solution {
    /**
     * 在满足 {@code richer} 拓扑序列的前提下，找到最安静的<BR>
     */
    public int[] loudAndRich(int[][] richer, int[] quiet) {
        int n = quiet.length;
        List<Integer>[] adj = new List[n];
        Arrays.setAll(adj, value -> new ArrayList<>());
        int[] inDegree = new int[n];
        for (int[] edge : richer) {
            // 有钱的 指向 没钱的
            int big = edge[0], small = edge[1];
            adj[big].add(small);
            ++inDegree[small];
        }
        // 找到入度为0的节点
        Deque<Integer> deque = new ArrayDeque<>(n);
        for (int i = 0; i < n; i++) {
            if (inDegree[i] == 0) {
                deque.addLast(i);
            }
        }
        int[] ans = new int[n];
        // 初始化每个节点的最安静的人是他自己
        for (int i = 0; i < n; ++i) {
            ans[i] = i;
        }
        // 拓扑排序
        while (!deque.isEmpty()) {
            int index = deque.pollFirst();
            // remove
            inDegree[index] = -1;
            for (Integer v : adj[index]) {
                if (--inDegree[v] == 0) {
                    deque.addLast(v);
                }
                // 如果以index为终点的链上有更安静的，就更新
                if (quiet[ans[v]] > quiet[ans[index]]) {
                    ans[v] = ans[index];
                }
            }
        }
        return ans;
    }
}
```

### 954. 二倍数对数组

题目链接：[954. 二倍数对数组](https://leetcode.cn/problems/array-of-doubled-pairs/)

时间复杂度：$O(n\log n)$

空间复杂度：$O(n)$

从绝对值小的开始寻找匹配，每次匹配掉它的两倍的数，如果没有足够的数和它匹配则返回 `false`

```java
public class Solution {
    public boolean canReorderDoubled(int[] arr) {
        HashMap<Integer, Integer> count = new HashMap<>(arr.length);
        for (int v : arr) {
            count.merge(v, 1, Integer::sum);
        }
        List<Integer> nums = new ArrayList<>(count.size());
        count.forEach((key, value) -> nums.add(key));
        nums.sort((o1, o2) -> Math.abs(o1) - Math.abs(o2));
        for (int num : nums) {
            int cnt = count.get(num);
            if (count.getOrDefault(2 * num, 0) < cnt) {
                return false;
            }
            count.merge(2 * num, -cnt, Integer::sum);
        }
        return true;
    }
}
```

### 2127. 参加会议的最多员工数

题目链接：[2127. 参加会议的最多员工数](https://leetcode.cn/problems/maximum-employees-to-be-invited-to-a-meeting/description/)

时间复杂度：$O(n)$

空间复杂度：$O(n)$

---

为了方便描述进行如下定义：

- 外向环：一个循环依赖的环，如果有一个或多个节点可以指向其他节点，则这个环是外向的
- 内向环：一个循环依赖的环，如果没有节点可以指向其他节点，则这个环是内向的

根据题目要求，每个节点和两个节点相连，或只有两个节点

该题出边有且只有一条，因此只有内向环（可能没有）

题目可以转化为求以下两种情况的最大值：

1. 尺寸大于2的环（这个环只能自己为一桌）
2. 多个尺寸等于2的内向环 或（尺寸为2的内向环+指向该环的链）

---

如何找到循环依赖的环呢？

通过拓扑排序找到循环依赖的环（逐个将入度为0的点删去，剩下的就是循环依赖的环）

---

对于尺寸为2的内向环，如何找到指向它的链呢？

从环的任意节点出发，对将每条入边转化为出边，一直走

由于该题出边有且只有一条，反向建图后入边有且只有一条，因此这一定不会走到另一个环

```java
public class Problem2127 {
    public int maximumInvitations(int[] favorite) {
        int n = favorite.length;
        // 计算入度
        int[] inDegree = new int[n];
        for (int v : favorite) {
            ++inDegree[v];
        }
        /*
          deque : 入度为0的点的下标
          inIndex[i] : 指向 i 的节点下标（反向建图）
         */
        Deque<Integer> deque = new ArrayDeque<>(n);
        List<Integer>[] inIndex = new List[n];
        Arrays.setAll(inIndex, value -> new ArrayList<>());
        for (int i = 0; i < n; ++i) {
            if (inDegree[i] == 0) {
                deque.addLast(i);
            }
            inIndex[favorite[i]].add(i);
        }
        // 拓扑排序 删点
        while (!deque.isEmpty()) {
            // 该图的每一个点均只有一条出边，即为 index -> favorite[index]
            int outIndex = favorite[deque.pollFirst()];
            if (--inDegree[outIndex] == 0) {
                deque.addLast(outIndex);
            }
        }
        /*
            ring2Size : 尺寸为2的环及其最长链的总节点数
            ringBigSize : 最大的尺寸大于2的环
         */
        int ring2Size = 0;
        int ringBigSize = 0;

        // 确定环
        for (int start = 0; start < n; start++) {
            if (inDegree[start] == 0) {
                continue;
            }
            // 标记环上的节点，避免重复求环
            inDegree[start] = 0;
            // 遍历环，确定环尺寸
            int size = 1;
            for (int next = favorite[start]; next != start; next = favorite[next]) {
                inDegree[next] = 0;
                ++size;
            }
            if (size == 2) {
                ring2Size += findChainMaxSize(start, favorite[start], inIndex) + findChainMaxSize(favorite[start], start, inIndex);
            } else {
                ringBigSize = Math.max(ringBigSize, size);
            }
        }
        return Math.max(ringBigSize, ring2Size);
    }

    /**
     * 找到指向start的链（反向建图的情况下）的最大长度，且不是通过other指向的
     *
     * @return 最大的链长度（包括出发点start）
     */
    private int findChainMaxSize(int start, int other, List<Integer>[] adj) {
        int max = 0;
        for (Integer v : adj[start]) {
            if (v != other) {
                max = Math.max(max, findChainMaxSize(v, other, adj));
            }
        }
        return max + 1;
    }
}
```

### 778. 水位上升的泳池中游泳

题目链接：[778. 水位上升的泳池中游泳](https://leetcode.cn/problems/swim-in-rising-water/)

#### 方法一

时间复杂度：$O(n^2\log n)$

空间复杂度：$O(n^2)$

题意：从左上角到右下角的路径中，找到路径值最小的值，其中路径值是路径上的所需要的最大时间

时间越多越可能走到终点，符合单调性，二分查找使得能从左上角到达右下角的最小的时间

```java
class Solution {
    /**
     * DISTANCE[i] -> {row, col}
     * left, up, right, down
     */
    private static final int[][] DISTANCE = {{0, -1}, {-1, 0}, {0, 1}, {1, 0}};

    private int n;
    private int[][] grid;

    public int swimInWater(int[][] grid) {
        this.n = grid.length;
        this.grid = grid;
        // 二分查找，找到使得能从左上角到达右下角的最小的时间
        int left = 0, right = n * n - 1;
        while (left < right) {
            int mid = left + right >> 1;
            if (grid[0][0] <= mid && dfs(0, 0, new boolean[n][n], mid)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    /**
     * @return 在时间为time的情况下，能否从(0,0)走到(n-1,n-1)
     */
    private boolean dfs(int row, int col, boolean[][] visited, int time) {
        visited[row][col] = true;
        if (row == n - 1 && col == n - 1) {
            return true;
        }
        for (int[] offset : DISTANCE) {
            int x = row + offset[0], y = col + offset[1];
            if (isLegal(x, y) && !visited[x][y] && grid[x][y] <= time && dfs(x, y, visited, time)) {
                return true;
            }
        }
        return false;
    }

    private boolean isLegal(int x, int y) {
        return 0 <= x && x < n && 0 <= y && y < n;
    }
}
```

#### 方法二

时间复杂度：$O(n^2\log n)$

空间复杂度：$O(n^2)$

由于每个格子所需要的时间是不同的，从小到大枚举时间 $t$，将所需时间为 $t$ 的格子染色，如果染色后左上角和右下角能够连通，则此时就是所需的最小时间，使用[并查集](####SimpleUnionFind)判断是否连通（将二维坐标转化为一维数字）

如果将矩阵看作图，两个相邻点之间有一条边权为两个节点的最大值的无向边，上述操作可以解释为每次选择边权最小的边，将这条边的两个点选择进入结果集合中，这正是 [Kruskal](#####Kruskal) 求解最小生成树的思想（学习算法的思维，而不是固定模式）

```java
class Solution2 {
    /**
     * DISTANCE[i] -> {row, col}
     * left, up, right, down
     */
    private static final int[][] DISTANCE = {{0, -1}, {-1, 0}, {0, 1}, {1, 0}};
    private int n;

    public int swimInWater(int[][] grid) {
        this.n = grid.length;
        /*
            index[i] : 所需时间为 i 的格子下标（转化为一维数字）
         */
        int len = n * n;
        int[] index = new int[len];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                index[grid[i][j]] = getIndex(i, j);
            }
        }
        SimpleUnionFind dsu = new SimpleUnionFind(len);
        for (int i = 0; i < len; i++) {
            int x = index[i] / n, y = index[i] % n;
            for (int[] offset : DISTANCE) {
                int nx = x + offset[0], ny = y + offset[1];
                if (isLegal(nx, ny) && grid[nx][ny] <= i) {
                    dsu.merge(index[i], getIndex(nx, ny));
                    if (dsu.same(0, len - 1)) {
                        return i;
                    }
                }
            }
        }
        return 0;
    }

    private int getIndex(int row, int col) {
        return row * n + col;
    }

    private boolean isLegal(int x, int y) {
        return 0 <= x && x < n && 0 <= y && y < n;
    }
}
```

#### 方法三

时间复杂度：$O(n^2\log n^2)$

空间复杂度：$O(n^2)$

按照方法二中所描述的，将矩阵看作图，相邻节点间有一条边权为两个节点的最大值的无向边，题目就是要求从左上角那个节点到右下角那个节点的“最优路径”，其中最优路径指的是路径上的最大值最小的一条路径。

仿照 [Dijkstra](#####Dijkstra) 求解最短路径（或 [Prim](#####Prim)求解最小生成树）的思路，每次选择一个“最优点”加入已选择部分，对新加入的点向外扩充

时间复杂度判断思路与Dijkstra或Prim相同

```java
class Solution3 {
    /**
     * DISTANCE[i] -> {row, col}
     * left, up, right, down
     */
    private static final int[][] DISTANCE = {{0, -1}, {-1, 0}, {0, 1}, {1, 0}};

    private int n;

    public int swimInWater(int[][] grid) {
        this.n = grid.length;
        int len = n * n;
        PriorityQueue<Node> queue = new PriorityQueue<>(len);
        boolean[][] visited = new boolean[n][n];
        visited[0][0] = true;
        int newestX = 0, newestY = 0;
        // 只需要一个参数即可记录最大值，因为对于无效分支的结果一定满足小于最后结果
        int max = 0;
        for (int k = 1; k < len; ++k) {
            // 从新加入的点向外扩充
            for (int[] offset : DISTANCE) {
                int x = newestX + offset[0], y = newestY + offset[1];
                if (isLegal(x, y)) {
                    int weight = Math.max(grid[newestX][newestY], grid[x][y]);
                    queue.add(new Node(x, y, weight));
                }
            }
            Node next = null;
            while (!queue.isEmpty()) {
                Node node = queue.poll();
                if (!visited[node.x][node.y]) {
                    next = node;
                    break;
                }
            }
            // 不可能出现
            if (next == null) {
                break;
            }
            // 将其加入选择部分
            newestX = next.x;
            newestY = next.y;
            visited[next.x][next.y] = true;
            max = Math.max(max, next.weight);
            // 如果终点移动至选择部分，则说明答案找到
            if (next.x == n - 1 && next.y == n - 1) {
                return max;
            }
        }
        return 0;
    }

    private boolean isLegal(int x, int y) {
        return 0 <= x && x < n && 0 <= y && y < n;
    }

    private static class Node implements Comparable<Node> {
        int x, y, weight;

        public Node(int x, int y, int weight) {
            this.x = x;
            this.y = y;
            this.weight = weight;
        }

        @Override
        public int compareTo(Node o) {
            return Integer.compare(weight, o.weight);
        }
    }
}
```

### 1631. 最小体力消耗路径

题目链接：[1631. 最小体力消耗路径](https://leetcode.cn/problems/path-with-minimum-effort/description/)

时间复杂度：$O()$

空间复杂度：$O()$

题目与 [778. 水位上升的泳池中游泳](###778. 水位上升的泳池中游泳)类似，仅边权变为高度差，求解过程与其相同

```java
class Solution {
    /**
     * DISTANCE[i] -> {row, col}
     * left, up, right, down
     */
    private static final int[][] DISTANCE = {{0, -1}, {-1, 0}, {0, 1}, {1, 0}};
    private int n, m;

    public int minimumEffortPath(int[][] heights) {
        this.n = heights.length;
        this.m = heights[0].length;
        int len = n * m;
        PriorityQueue<Node> queue = new PriorityQueue<>(len);
        boolean[][] visited = new boolean[n][m];
        visited[0][0] = true;
        int newestX = 0, newestY = 0;
        int max = 0;
        for (int k = 1; k < len; ++k) {
            // 从新加入的点向外扩充
            for (int[] offset : DISTANCE) {
                int x = newestX + offset[0], y = newestY + offset[1];
                if (isLegal(x, y)) {
                    int weight = Math.abs(heights[newestX][newestY] - heights[x][y]);
                    queue.add(new Node(x, y, weight));
                }
            }
            Node next = null;
            while (!queue.isEmpty()) {
                Node node = queue.poll();
                if (!visited[node.x][node.y]) {
                    next = node;
                    break;
                }
            }
            // 不可能出现
            if (next == null) {
                break;
            }
            // 将其加入选择部分
            newestX = next.x;
            newestY = next.y;
            visited[next.x][next.y] = true;
            max = Math.max(max, next.weight);
            // 如果终点移动至选择部分，则说明答案找到
            if (next.x == n - 1 && next.y == m - 1) {
                return max;
            }
        }
        return 0;
    }

    private boolean isLegal(int row, int col) {
        return 0 <= row && row < n && 0 <= col && col < m;
    }

    private static class Node implements Comparable<Node> {
        int x, y, weight;

        public Node(int x, int y, int weight) {
            this.x = x;
            this.y = y;
            this.weight = weight;
        }

        @Override
        public int compareTo(Node o) {
            return Integer.compare(weight, o.weight);
        }
    }
}
```

### 743. 网络延迟时间

题目链接：[743. 网络延迟时间](https://leetcode.cn/problems/network-delay-time/description/)

时间复杂度：$O(n^2+m)$，其中，$m$ 为`times.length`

空间复杂度：$O(n+m)$

求单源最短路，然后求所有最短路中的最大值

```java
class Solution {
    public int networkDelayTime(int[][] times, int n, int k) {
        List<Node>[] adj = new List[n];
        Arrays.setAll(adj, value -> new ArrayList<>());
        for (int[] time : times) {
            int from = time[0] - 1, to = time[1] - 1, weight = time[2];
            adj[from].add(new Node(to, weight));
        }
        int[] distances = new Dijkstra().getSingleSourceShortPath(k - 1, adj);
        int ans = 0;
        for (int distance : distances) {
            if (distance == Dijkstra.INF) {
                return -1;
            }
            ans = Math.max(ans, distance);
        }
        return ans;
    }
}
```

### 2304. 网格中的最小路径代价

题目链接：[2304. 网格中的最小路径代价](https://leetcode.cn/problems/minimum-path-cost-in-a-grid/description)

时间复杂度：$O(n^2m)$

空间复杂度：$O(nm)$

思路类似于[数字三角形](https://www.luogu.com.cn/problem/T291913)

```java
// 记忆化写法
class Solution1 {
    private static final int CACHE_INIT = -1;
    private static final int INF = Integer.MAX_VALUE;
    int n, m;
    int[][] grid, moveCost;
    int[][] cache;

    public int minPathCost(int[][] grid, int[][] moveCost) {
        n = grid.length;
        m = grid[0].length;
        this.grid = grid;
        this.moveCost = moveCost;
        this.cache = new int[n][m];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(cache[i], CACHE_INIT);
        }
        int ans = INF;
        for (int i = 0; i < m; ++i) {
            ans = Math.min(ans, function(n - 1, i));
        }
        return ans;
    }

    /**
     * f(x,y) = grid[x][y] + min( f(x-1,...)+... )
     */
    private int function(int x, int y) {
        if (cache[x][y] != CACHE_INIT) {
            return cache[x][y];
        }
        if (x == 0) {
            return cache[x][y] = grid[x][y];
        }
        int ans = INF;
        for (int beforeY = 0; beforeY < m; ++beforeY) {
            int beforeValue = grid[x - 1][beforeY];
            ans = Math.min(ans, function(x - 1, beforeY) + moveCost[beforeValue][y]);
        }
        ans += grid[x][y];
        cache[x][y] = ans;
        return ans;
    }
}

// 循环写法
class Solution2 {
    public int minPathCost(int[][] grid, int[][] moveCost) {
        int n = grid.length, m = grid[0].length;
        int[][] dp = new int[n][m];
        System.arraycopy(grid[0], 0, dp[0], 0, m);
        for (int i = 1; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                dp[i][j] = Integer.MAX_VALUE;
                // 枚举前一行
                for (int beforeY = 0; beforeY < m; beforeY++) {
                    int beforeValue = grid[i - 1][beforeY];
                    dp[i][j] = Math.min(dp[i][j], dp[i - 1][beforeY] + grid[i][j] + moveCost[beforeValue][j]);
                }
            }
        }
        return Arrays.stream(dp[n - 1]).min().getAsInt();
    }
}
```

## 附录

**以下类均未进行测试**

### DSU

#### SimpleUnionFind

```java
/**
 * @author : Cattle_Horse
 * @date : 2023/11/11 13:36
 * @description : 简单并查集
 **/
class SimpleUnionFind {
    /**
     * initialSize : 初始化集合个数
     * currentSize : 当前集合个数
     * nodeFathers : 父节点
     */
    protected final int initialSize;
    protected int currentSize;
    protected final int[] nodeFathers;

    public SimpleUnionFind(int initialSize) {
        this.initialSize = initialSize;
        this.currentSize = initialSize;
        nodeFathers = new int[initialSize];
        for (int i = 0; i < initialSize; ++i) {
            nodeFathers[i] = i;
        }
    }

    public int getInitialSize() {
        return initialSize;
    }

    public int getCurrentSize() {
        return currentSize;
    }

    public int[] getNodeFathers() {
        return nodeFathers;
    }

    /**
     * 判断 id 是否符合要求
     *
     * @param id id
     * @return 是否符合要求
     */
    public boolean nonstandardId(int id) {
        return id < 0 || id >= initialSize;
    }

    /**
     * 查找元素所属集合根节点编号
     *
     * @param id 待查找元素
     * @return 所属集合的根节点编号
     */
    public int find(int id) {
        if (nonstandardId(id)) {
            throw new IllegalArgumentException(String.format("id should be in the range [%d, %d), but id is %d", 0, initialSize, id));
        }
        return findUncheck(id);
    }

    /**
     * 查找元素所属集合根节点编号（不进行id检查）
     *
     * @param id 待查找元素
     * @return 所属集合的根节点编号
     */
    private int findUncheck(int id) {
        if (nodeFathers[id] == id) {
            return id;
        }
        return nodeFathers[id] = find(nodeFathers[id]);
        // while (id != nodeFathers[id]) {
        //     nodeFathers[id] = nodeFathers[nodeFathers[id]];
        //     id = nodeFathers[id];
        // }
        // return id;
    }

    /**
     * 判断两个元素是否属于同一个集合
     *
     * @param x 元素1
     * @param y 元素2
     * @return 是否属于同一个集合
     */
    public boolean same(int x, int y) {
        return find(x) == find(y);
    }

    /**
     * 合并两个集合，并返回是否合并成功
     * 将集合x合并至集合y中
     *
     * @param x 集合1
     * @param y 集合2
     * @return 是否合并成功
     */
    public boolean merge(int x, int y) {
        int fx = find(x), fy = find(y);
        if (fx == fy) {
            return false;
        }
        nodeFathers[fx] = fy;
        --currentSize;
        return true;
    }
}
```

#### UnionFind

```java
/**
 * @author : Cattle_Horse
 * @date : 2023/11/4 15:17
 * @description : 启发式合并并查集
 **/
public class UnionFind extends SimpleUnionFind {
    /**
     * nodeCounts  : 集合中节点个数
     */
    private final int[] nodeCounts;

    public UnionFind(int initialSize) {
        super(initialSize);
        this.currentSize = initialSize;
        nodeCounts = new int[initialSize];
        for (int i = 0; i < initialSize; ++i) {
            nodeCounts[i] = 1;
        }
    }

    public int getNodeCount(int id) {
        return nodeCounts[find(id)];
    }

    public int[] getNodeCounts() {
        return nodeCounts;
    }

    /**
     * 合并两个集合，并返回是否合并成功
     *
     * @param x 集合1
     * @param y 集合2
     * @return 是否合并成功
     */
    @Override
    public boolean merge(int x, int y) {
        int fx = find(x), fy = find(y);
        if (fx == fy) {
            return false;
        }
        // 如果fx的节点个数小于fy，则将fx合并到fy中，否则将fy合并到fx中
        if (nodeCounts[fx] < nodeCounts[fy]) {
            nodeFathers[fx] = fy;
            nodeCounts[fy] += nodeCounts[fx];
        } else {
            nodeFathers[fy] = fx;
            nodeCounts[fx] += nodeCounts[fy];
        }
        --currentSize;
        return true;
    }
}
```

### graph

#### base

##### Node

```java
/**
 * @author : Cattle_Horse
 * @date : 2023/12/5 18:58
 * @description : 绑定to和weight，用于Dijkstra,Prim与Adjacency
 **/
class Node implements Comparable<Node> {
    public int to;
    public int weight;

    public Node(int to, int weight) {
        this.to = to;
        this.weight = weight;
    }

    @Override
    public int compareTo(Node o) {
        return Integer.compare(weight, o.weight);
    }
}
```

##### Edge

```java
/**
 * @author : Cattle_Horse
 * @date : 2023/12/2 17:43
 * @description : 边
 **/
class Edge implements Comparable<Edge> {
    /**
     * 一条从from到to，权重为weight 的有向边
     */
    public int from, to;
    public int weight;

    public Edge(int from, int to, int weight) {
        this.from = from;
        this.to = to;
        this.weight = weight;
    }

    @Override
    public int compareTo(Edge o) {
        return Integer.compare(weight, o.weight);
    }
}
```

##### Graph

```java
/**
 * @author : Cattle_Horse
 * @date : 2023/12/1 18:26
 * @description : 有向图
 **/
abstract class Graph {
    public static final int INF = 0x3f3f3f3f;
    /**
     * vertexCount : 图的点数
     * edgeCount : 图的边数
     */
    protected int vertexCount;
    protected int edgeCount;

    public Graph(int vertexCount) {
        this.vertexCount = vertexCount;
        this.edgeCount = 0;
    }

    public int getVertexCount() {
        return vertexCount;
    }

    public int getEdgeCount() {
        return edgeCount;
    }

    /**
     * 入度
     *
     * @return 入度
     */
    public abstract int[] getInDegree();

    /**
     * 出度
     *
     * @return 出度
     */
    public abstract int[] getOutDegree();

    /**
     * 添加一条从from到to 且 权重为weight 的有向边
     *
     * @param from   起始点
     * @param to     终点
     * @param weight 权重
     */
    public abstract void addEdge(int from, int to, int weight);

    /**
     * 添加一条有向边
     *
     * @param edge 有向边
     */
    public abstract void addEdge(Edge edge);

    /**
     * 获得图的边集
     * @return 边集
     */
    public abstract List<Edge> getEdges();

    /**
     * 最小生成树
     *
     * @param mst 使用的最小生成树算法
     * @return 最小生成树的边权和
     * @see MinimumSpanningTree
     */
    public abstract int getMinimumSpanningTree(MinimumSpanningTree mst);

    /**
     * 单源最短路径长度
     *
     * @param start 源点
     * @param sp    使用的最短路径算法
     * @return 从 start 作为起点到达其余各点的最短路径长度
     * @see ShortestPath
     */
    public abstract int[] getShortestPath(int start, ShortestPath sp);

    /**
     * 全源最短路径长度
     *
     * @param sp 使用的最短路径算法
     * @return result[x][y]表示从x到y的最短路径长度
     * @see ShortestPath
     */
    public abstract int[][] getShortestPath(ShortestPath sp);

    /**
     * 拓扑排序
     *
     * @param result          排序后的拓扑序列存储位置（需满足长度大于等于图的点数）
     * @param topologicalSort 使用的拓扑排序算法
     * @return 是否存在拓扑排序
     * @see TopologicalSort
     */
    public abstract boolean topologicalSort(int[] result, TopologicalSort topologicalSort);
}
```

##### Matrix

```java
/**
 * @author : Cattle_Horse
 * @date : 2023/12/2 17:20
 * @description : 邻接矩阵
 **/
class Matrix extends Graph {
    private final int[][] mat;

    public Matrix(int vertexCount) {
        super(vertexCount);
        mat = new int[vertexCount][vertexCount];
        for (int i = 0; i < vertexCount; i++) {
            Arrays.fill(mat[i], INF);
            mat[i][i] = 0;
        }
    }

    public int[][] getMat() {
        return mat;
    }

    public int[] getNeighbor(int id) {
        return mat[id];
    }

    @Override
    public int[] getInDegree() {
        int[] inDegree = new int[vertexCount];
        for (int from = 0; from < vertexCount; from++) {
            for (int to = 0; to < vertexCount; to++) {
                if (mat[from][to] != INF) {
                    ++inDegree[to];
                }
            }
        }
        return inDegree;
    }

    @Override
    public int[] getOutDegree() {
        int[] outDegree = new int[vertexCount];
        for (int from = 0; from < vertexCount; from++) {
            for (int weight : mat[from]) {
                if (weight != INF) {
                    ++outDegree[from];
                }
            }
        }
        return outDegree;
    }

    @Override
    public void addEdge(int from, int to, int weight) {
        ++edgeCount;
        mat[from][to] = weight;
    }

    @Override
    public void addEdge(Edge edge) {
        this.addEdge(edge.from, edge.to, edge.weight);
    }

    @Override
    public List<Edge> getEdges() {
        List<Edge> edges = new ArrayList<>(edgeCount);
        for (int i = 0; i < vertexCount; i++) {
            for (int j = 0; j < vertexCount; j++) {
                if (mat[i][j] != INF && i != j) {
                    edges.add(new Edge(i, j, mat[i][j]));
                }
            }
        }
        return edges;
    }

    @Override
    public int getMinimumSpanningTree(MinimumSpanningTree mst) {
        return mst.apply(mat, INF);
    }

    @Override
    public int[] getShortestPath(int start, ShortestPath sp) {
        return sp.getSingleSourceShortPath(start, mat, INF);
    }

    @Override
    public int[][] getShortestPath(ShortestPath sp) {
        return sp.getFullSourceShortPath(mat, INF);
    }

    @Override
    public boolean topologicalSort(int[] result, TopologicalSort topologicalSort) {
        return topologicalSort.apply(result, mat, INF);
    }
}
```

##### Adjacency

```java
/**
 * @author : Cattle_Horse
 * @date : 2023/12/2 17:19
 * @description : 邻接表
 **/
class Adjacency extends Graph {
    private final List<Node>[] adj;

    public Adjacency(int vertexCount) {
        super(vertexCount);
        adj = new ArrayList[vertexCount];
        Arrays.setAll(adj, value -> new ArrayList<>());
    }

    public List<Node>[] getAdj() {
        return adj;
    }

    public List<Node> getNeighbor(int id) {
        return adj[id];
    }

    @Override
    public int[] getInDegree() {
        int[] inDegree = new int[vertexCount];
        for (int from = 0; from < vertexCount; from++) {
            for (Node node : adj[from]) {
                ++inDegree[node.to];
            }
        }
        return inDegree;
    }

    @Override
    public int[] getOutDegree() {
        int[] outDegree = new int[vertexCount];
        for (int from = 0; from < vertexCount; from++) {
            outDegree[from] = adj[from].size();
        }
        return outDegree;
    }

    @Override
    public void addEdge(int from, int to, int weight) {
        ++edgeCount;
        adj[from].add(new Node(to, weight));
    }

    @Override
    public void addEdge(Edge edge) {
        this.addEdge(edge.from, edge.to, edge.weight);
    }

    @Override
    public List<Edge> getEdges() {
        List<Edge> edges = new ArrayList<>(edgeCount);
        for (int i = 0; i < vertexCount; i++) {
            for (Node node : adj[i]) {
                edges.add(new Edge(i, node.to, node.weight));
            }
        }
        return edges;
    }

    @Override
    public int getMinimumSpanningTree(MinimumSpanningTree mst) {
        return mst.apply(adj);
    }

    @Override
    public int[] getShortestPath(int start, ShortestPath sp) {
        return sp.getSingleSourceShortPath(start, adj);
    }

    @Override
    public int[][] getShortestPath(ShortestPath sp) {
        return sp.getFullSourceShortPath(adj);
    }

    @Override
    public boolean topologicalSort(int[] result, TopologicalSort topologicalSort) {
        return topologicalSort.apply(result, adj);
    }
}
```

#### mst

##### MinimumSpanningTree

```java
/**
 * @author : Cattle_Horse
 * @date : 2023/12/5 18:47
 * @description : 最小生成树接口
 **/
interface MinimumSpanningTree {
    int INF = 0x3f3f3f3f;

    /**
     * 最小生成树
     *
     * @param matrix 待拓扑排序的邻接矩阵
     * @param inf    标志矩阵中不存在边的值
     * @return 最小生成树边权和，如果图不连通则返回MinimumSpanningTree.INF
     */
    int apply(int[][] matrix, final int inf);

    /**
     * 最小生成树
     *
     * @param adj 待求解的邻接表
     * @return 最小生成树边权和，如果图不连通则返回MinimumSpanningTree.INF
     */
    int apply(List<Node>[] adj);
}
```

##### Kruskal

```java
/**
 * @author : Cattle_Horse
 * @date : 2023/12/5 22:06
 * @description : Kruskal求最小生成树<BR>
 * 把所有边的权重从小到大排列，只要不形成环（可以使用并查集判断环）就加入到最终的生成树中<BR>
 * <ul>
 *  其中 V 是图中顶点个数，E 是边数
 *  <li>若直接传入边集 {@code apply} 方法的时间复杂度为 {@code O( E \log E )}</li>
 *  <li>若邻接表或邻接矩阵 {@code apply} 方法的时间复杂度为 {@code O( E \log E + V^2)}</li>
 * </ul>
 * @see MinimumSpanningTree
 **/
class Kruskal implements MinimumSpanningTree {
    /**
     * @param n     节点数
     * @param edges 边集
     * @return 最小生成树边权和，如果图不连通则返回 MinimumSpanningTree.INF
     * @see SimpleUnionFind
     */
    public int apply(int n, List<Edge> edges) {
        SimpleUnionFind dsu = new SimpleUnionFind(n);
        Collections.sort(edges);
        int sum = 0;
        int size = 0;
        for (Edge edge : edges) {
            if (dsu.merge(edge.from, edge.to)) {
                sum += edge.weight;
                ++size;
                if (size == n - 1) {
                    break;
                }
            }
        }
        return size == n - 1 ? sum : INF;
    }

    @Override
    public int apply(int[][] matrix, int inf) {
        List<Edge> edges = new ArrayList<>();
        int n = matrix.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] != inf && i != j) {
                    edges.add(new Edge(i, j, matrix[i][j]));
                }
            }
        }
        return this.apply(n, edges);
    }

    @Override
    public int apply(List<Node>[] adj) {
        List<Edge> edges = new ArrayList<>();
        int n = adj.length;
        for (int i = 0; i < n; i++) {
            for (Node node : adj[i]) {
                edges.add(new Edge(i, node.to, node.weight));
            }
        }
        return this.apply(n, edges);
    }
}
```

##### Prim

```java
/**
 * @author : Cattle_Horse
 * @date : 2023/12/5 21:22
 * @description : 朴素的Prim算法求最小生成树<BR>
 * 将图分为已选择的点和未选择的点两部分，每次扩充一个距离已选择的部分最近的点<BR>
 * {@code apply} 方法的时间复杂度为 {@code O( V^2 )}，其中 V 是图中顶点个数
 * @see MinimumSpanningTree
 **/
class Prim implements MinimumSpanningTree {
    /**
     * CHOSEN : 表示节点是否被选择，设置为负无穷，在update时少判断一次
     */
    private static final int CHOSEN = Integer.MIN_VALUE;
    private static final int NONE = -1;

    /**
     * 在distance中,找到distance[index]最小的 且{@code distance[index] != CHOSEN} 的 index，如果不存在则返回 {@code NONE}
     */
    private static int findMinNode(int[] distance) {
        int min = Integer.MAX_VALUE, index = NONE;
        for (int i = 0, n = distance.length; i < n; i++) {
            if (distance[i] != CHOSEN && distance[i] < min) {
                min = distance[i];
                index = i;
            }
        }
        return index;
    }

    @Override
    public int apply(int[][] matrix, int inf) {
        /*
            n : 节点个数
            newest : 目前选择的最新的节点
            sum : 最小生成树权值和
            visit : visit[i]表示i节点是否选择（在distance中使用特殊值表示是否选择，减少内存消耗）
            distance : distance[i]表示从已选择的点中任意一个点作为起点到达i的最短距离（若为CHOSEN则表示已选择）
         */
        int n = matrix.length;
        int newest = 0;
        int sum = 0;
        int[] distance = new int[n];
        Arrays.fill(distance, Integer.MAX_VALUE);
        distance[newest] = CHOSEN;
        // 再选n-1个点
        for (int k = 1; k < n; ++k) {
            // 1. Update 对新加入的点进行延申
            for (int to = 0; to < n; ++to) {
                // distance[to] != CHOSEN
                if (matrix[newest][to] != inf && distance[to] > matrix[newest][to]) {
                    distance[to] = matrix[newest][to];
                }
            }
            // 2. Scan 找到连接已选择部分和未选择部分的最小边
            int next = findMinNode(distance);
            // 没有可以继续延申的点了，剩下的是不可达的
            if (next == NONE) {
                break;
            }
            // 3. Add 加入已选择顶点部分
            newest = next;
            sum += distance[newest];
            distance[newest] = CHOSEN;
        }
        return sum;
    }

    @Override
    public int apply(List<Node>[] adj) {
       /*
            n : 节点个数
            newest : 目前选择的最新的节点
            sum : 最小生成树权值和
            visit : visit[i]表示i节点是否选择（在distance中使用特殊值表示是否选择，减少内存消耗）
            distance : distance[i]表示从已选择的点中任意一个点作为起点到达i的最短距离（若为CHOSEN则表示已选择）
         */
        int n = adj.length;
        int newest = 0;
        int sum = 0;
        int[] distance = new int[n];
        Arrays.fill(distance, Integer.MAX_VALUE);
        distance[newest] = CHOSEN;
        // 再选n-1个点
        for (int k = 1; k < n; ++k) {
            // 1. Update 对新加入的点进行延申
            for (Node node : adj[newest]) {
                // distance[node.to] != CHOSEN
                if (distance[node.to] > node.weight) {
                    distance[node.to] = node.weight;
                }
            }
            // 2. Scan 找到连接已选择部分和未选择部分的最小边
            int next = findMinNode(distance);
            // 没有可以继续延申的点了，剩下的是不可达的
            if (next == NONE) {
                break;
            }
            // 3. Add 加入已选择顶点部分
            newest = next;
            sum += distance[newest];
            distance[newest] = CHOSEN;
        }
        return sum;
    }
}
```

##### HeapPrim

```java
/**
 * @author : Cattle_Horse
 * @date : 2023/12/5 21:53
 * @description : 堆优化的Prim算法求最小生成树<BR>
 * 将图分为已选择的点和未选择的点两部分，每次扩充一个距离已选择的部分最近的点<BR>
 * {@code apply} 方法的时间复杂度为 {@code O( (V+E) \log V )}，其中 V 是图中顶点个数，E是边数
 * @see MinimumSpanningTree
 **/
class HeapPrim implements MinimumSpanningTree {
    /**
     * 在queue中,找到最小的 {@code visit[node.to] == false} 的 {@code Node}，如果不存在则返回 {@code null}
     */
    private static Node findMinNode(PriorityQueue<Node> queue, boolean[] visit) {
        while (!queue.isEmpty()) {
            Node node = queue.poll();
            if (!visit[node.to]) {
                return node;
            }
        }
        return null;
    }

    @Override
    public int apply(int[][] matrix, int inf) {
        /*
            n : 节点个数
            newest : 目前选择的最新的节点
            sum : 最小生成树权值和
            queue : 参照 朴素的Prim算法 中distance的定义，但是这个queue时从小到大排序的
            visit : visit[i]表示i节点是否选择
         */
        int n = matrix.length;
        int newest = 0;
        int sum = 0;
        PriorityQueue<Node> queue = new PriorityQueue<>(n);
        boolean[] visit = new boolean[n];
        visit[newest] = true;
        // 再选n-1个点
        for (int k = 1; k < n; ++k) {
            // 1. Update 对新加入的点进行延申
            for (int to = 0; to < n; ++to) {
                if (matrix[newest][to] != inf && !visit[to]) {
                    queue.add(new Node(to, matrix[newest][to]));
                }
            }
            // 2. Scan 找到连接已选择部分和未选择部分的最小边
            Node next = findMinNode(queue, visit);
            // 没有可以继续延申的点了，剩下的是不可达的
            if (next == null) {
                break;
            }
            // 3. Add 加入已选择顶点部分
            newest = next.to;
            sum += next.weight;
            visit[newest] = true;
        }
        return sum;
    }

    @Override
    public int apply(List<Node>[] adj) {
            /*
            n : 节点个数
            newest : 目前选择的最新的节点
            sum : 最小生成树权值和
            queue : 参照 朴素的Prim算法 中distance的定义，但是这个queue时从小到大排序的
            visit : visit[i]表示i节点是否选择
         */
        int n = adj.length;
        int newest = 0;
        int sum = 0;
        PriorityQueue<Node> queue = new PriorityQueue<>(n);
        boolean[] visit = new boolean[n];
        visit[newest] = true;
        // 再选n-1个点
        for (int k = 1; k < n; ++k) {
            // 1. Update 对新加入的点进行延申
            for (Node node : adj[newest]) {
                if (!visit[node.to]) {
                    // queue.add(new Node(node.to, node.weight));
                    queue.add(node);
                }
            }
            // 2. Scan 找到连接已选择部分和未选择部分的最小边
            Node next = findMinNode(queue, visit);
            // 没有可以继续延申的点了，剩下的是不可达的
            if (next == null) {
                break;
            }
            // 3. Add 加入已选择顶点部分
            newest = next.to;
            sum += next.weight;
            visit[newest] = true;
        }
        return sum;
    }
}
```

#### shortest

##### ShortestPath

```java
/**
 * @author : Cattle_Horse
 * @date : 2023/12/2 17:04
 * @description : 最短路接口
 **/
interface ShortestPath {
    int INF = 0x3f3f3f3f;

    /**
     * 单源最短路径长度
     *
     * @param start  源点
     * @param matrix 待拓扑排序的邻接矩阵
     * @param inf    标志矩阵中不存在边的值
     * @return 从 start 作为起点到达其余各点的最短路径长度，若不可达则返回 ShortestPath.INF
     */
    int[] getSingleSourceShortPath(int start, int[][] matrix, final int inf);

    /**
     * 单源最短路径长度
     *
     * @param start     源点
     * @param adjacency 待拓扑排序的邻接表
     * @return 从 start 作为起点到达其余各点的最短路径长度，若不可达则返回 ShortestPath.INF
     */
    int[] getSingleSourceShortPath(int start, List<Node>[] adjacency);

    /**
     * 全源最短路径长度
     *
     * @param matrix 待拓扑排序的邻接矩阵
     * @param inf    标志矩阵中不存在边的值
     * @return result[x][y]表示从x到y的最短路径长度，若不可达则返回 ShortestPath.INF
     */
    int[][] getFullSourceShortPath(int[][] matrix, final int inf);

    /**
     * 全源最短路径长度
     *
     * @param adjacency 待拓扑排序的邻接表
     * @return result[x][y]表示从x到y的最短路径长度，若不可达则返回 ShortestPath.INF
     */
    int[][] getFullSourceShortPath(List<Node>[] adjacency);
}
```

##### Floyd

```java
/**
 * @author : Cattle_Horse
 * @date : 2023/12/2 20:51
 * @description : Floyd算法求全源最短路<BR>
 * 判断图中任意两点的最短路径是否经过了某一点<BR>
 * {@code getSingleSourceShortPath} 与 {@code getFullSourceShortPath} 方法的时间复杂度均为 {@code O( V^3 )}，其中 V 是图中顶点个数，E 是边数
 * @see ShortestPath
 **/
class Floyd implements ShortestPath {
    /**
     * 在初始化二维距离矩阵后调用这个方法进行动态规划
     */
    public void floyd(int[][] distance, int n) {
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (distance[i][j] > distance[i][k] + distance[k][j]) {
                        distance[i][j] = distance[i][k] + distance[k][j];
                    }
                }
            }
        }
    }

    @Override
    public int[] getSingleSourceShortPath(int start, int[][] matrix, int inf) {
        return getFullSourceShortPath(matrix, inf)[start];
    }

    @Override
    public int[] getSingleSourceShortPath(int start, List<Node>[] adjacency) {
        return getFullSourceShortPath(adjacency)[start];
    }

    @Override
    public int[][] getFullSourceShortPath(int[][] matrix, int inf) {
        int n = matrix.length;
        int[][] distance = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                distance[i][j] = i == j ? 0 : (matrix[i][j] == inf ? INF : matrix[i][j]);
            }
        }
        floyd(distance, n);
        return distance;
    }

    @Override
    public int[][] getFullSourceShortPath(List<Node>[] adjacency) {
        int n = adjacency.length;
        int[][] distance = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                distance[i][j] = i == j ? 0 : INF;
            }
        }
        for (int i = 0; i < n; i++) {
            for (Node node : adjacency[i]) {
                distance[i][node.to] = node.weight;
            }
        }
        floyd(distance, n);
        return distance;
    }
}
```

##### Dijkstra

```java
/**
 * @author : Cattle_Horse
 * @date : 2023/12/5 20:31
 * @description : 朴素的Dijkstra求单源最短路<BR>
 * 将图分为已选择的点和未选择的点两部分，每次扩充一个"最短"的点<BR>
 * <ul>
 *  其中 V 是图中顶点个数，E 是边数
 *  <li>{@code getSingleSourceShortPath} 方法的时间复杂度为 {@code O( V^2 )}</li>
 *  <li>{@code getFullSourceShortPath} 方法的时间复杂度为 {@code O( V^3 )}</li>
 * </ul>
 * @see ShortestPath
 **/
class Dijkstra implements ShortestPath {
    private static final int NONE = -1;

    /**
     * 在distance中,找到distance[index]最小的 且{@code visit[index] == false} 的 index，如果不存在则返回 {@code NONE}
     */
    private static int findMinNode(int[] distance, boolean[] visit) {
        int min = Integer.MAX_VALUE, index = NONE;
        for (int i = 0, n = distance.length; i < n; i++) {
            if (!visit[i] && distance[i] < min) {
                min = distance[i];
                index = i;
            }
        }
        return index;
    }

    @Override
    public int[] getSingleSourceShortPath(int start, int[][] matrix, int inf) {
        int n = matrix.length;
        int[] distance = new int[n];
        Arrays.fill(distance, INF);
        // 是否已经选择过该点
        boolean[] visit = new boolean[n];
        // 最新添加的节点
        int newest = start;
        distance[newest] = 0;
        visit[newest] = true;
        // 再选n-1个点
        for (int k = 1; k < n; ++k) {
            // 1. Update 对新加入的点进行延申
            for (int i = 0; i < n; ++i) {
                if (matrix[newest][i] != inf && !visit[i] && distance[i] > distance[newest] + matrix[newest][i]) {
                    distance[i] = distance[newest] + matrix[newest][i];
                }
            }
            // 2. Scan 找到连接已选择部分和未选择部分的最小边
            int next = findMinNode(distance, visit);
            // 没有可以继续延申的点了，剩下的是不可达的
            if (next == NONE) {
                break;
            }
            // 3. Add 加入已选择顶点部分
            newest = next;
            visit[newest] = true;
        }
        return distance;
    }

    @Override
    public int[] getSingleSourceShortPath(int start, List<Node>[] adjacency) {
        int n = adjacency.length;
        int[] distance = new int[n];
        Arrays.fill(distance, INF);
        // 是否已经选择过该点
        boolean[] visit = new boolean[n];
        // 最新添加的节点
        int newest = start;
        distance[newest] = 0;
        visit[newest] = true;
        // 再选n-1个点
        for (int k = 1; k < n; ++k) {
            // 1. Update 对新加入的点进行延申
            for (Node node : adjacency[newest]) {
                if (!visit[node.to] && distance[node.to] > distance[newest] + node.weight) {
                    distance[node.to] = distance[newest] + node.weight;
                }
            }
            // 2. Scan 找到连接已选择部分和未选择部分的最小边
            int next = findMinNode(distance, visit);
            // 没有可以继续延申的点了，剩下的是不可达的
            if (next == NONE) {
                break;
            }
            // 3. Add 加入已选择顶点部分
            newest = next;
            visit[newest] = true;
        }
        return distance;
    }

    @Override
    public int[][] getFullSourceShortPath(int[][] matrix, int inf) {
        int n = matrix.length;
        int[][] distance = new int[n][];
        for (int i = 0; i < n; i++) {
            distance[i] = getSingleSourceShortPath(i, matrix, inf);
        }
        return distance;
    }

    @Override
    public int[][] getFullSourceShortPath(List<Node>[] adjacency) {
        int n = adjacency.length;
        int[][] distance = new int[n][];
        for (int i = 0; i < n; i++) {
            distance[i] = getSingleSourceShortPath(i, adjacency);
        }
        return distance;
    }
}
```

##### HeapDijkstra

```java
/**
 * @author : Cattle_Horse
 * @date : 2023/12/5 20:19
 * @description : 堆优化的Dijkstra求单源最短路<BR>
 * 将图分为已选择的点和未选择的点两部分，每次扩充一个"最短"的点<BR>
 * <ul>
 *  其中 V 是图中顶点个数，E 是边数
 *  <li>{@code getSingleSourceShortPath} 方法的时间复杂度为 {@code O( (E+V) \log V )}</li>
 *  <li>{@code getFullSourceShortPath} 方法的时间复杂度为 {@code O( V (E+V) \log V )}</li>
 * </ul>
 * @see ShortestPath
 **/
class HeapDijkstra implements ShortestPath {
    /**
     * 在queue中,找到最小的 {@code visit[node.to] == false} 的 {@code Node}，如果不存在则返回 {@code null}
     */
    private static Node findMinNode(PriorityQueue<Node> queue, boolean[] visit) {
        while (!queue.isEmpty()) {
            Node node = queue.poll();
            if (!visit[node.to]) {
                return node;
            }
        }
        return null;
    }

    @Override
    public int[] getSingleSourceShortPath(int start, int[][] matrix, int inf) {
        /*
            n : 节点个数
            distance : distance[i] 表示目前 start 到 i 的最短路径长度
            queue : 参照 朴素的Dijkstra算法 中distance的定义，但是这个queue时从小到大排序的
            visit : visit[i] 表示目前节点i是否被选择
            newest : 最新添加的节点
         */
        int n = matrix.length;
        int[] distance = new int[n];
        Arrays.fill(distance, INF);
        PriorityQueue<Node> queue = new PriorityQueue<>(n);
        boolean[] visit = new boolean[n];
        int newest = start;
        distance[newest] = 0;
        visit[newest] = true;
        // 再选n-1个点
        for (int k = 1; k < n; ++k) {
            // 1. Update 对新加入的点进行延申
            for (int i = 0; i < n; ++i) {
                if (matrix[newest][i] != inf && !visit[i] && distance[i] > distance[newest] + matrix[newest][i]) {
                    distance[i] = distance[newest] + matrix[newest][i];
                    queue.add(new Node(i, distance[i]));
                }
            }
            // 2. Scan 找到连接已选择部分和未选择部分的最小边
            Node next = findMinNode(queue, visit);
            // 没有可以继续延申的点了，剩下的是不可达的
            if (next == null) {
                break;
            }
            // 3. Add 加入已选择顶点部分
            newest = next.to;
            visit[newest] = true;
        }
        return distance;
    }

    @Override
    public int[] getSingleSourceShortPath(int start, List<Node>[] adjacency) {
         /*
            n : 节点个数
            distance : distance[i] 表示目前 start 到 i 的最短路径长度
            queue : 目前从start出发到其余点的距离（按照权值从小到大排序）
            visit : visit[i] 表示目前节点i是否被选择
            newest : 最新添加的节点
         */
        int n = adjacency.length;
        int[] distance = new int[n];
        Arrays.fill(distance, INF);
        PriorityQueue<Node> queue = new PriorityQueue<>(n);
        boolean[] visit = new boolean[n];
        int newest = start;
        distance[newest] = 0;
        visit[newest] = true;
        // 再选n-1个点
        for (int k = 1; k < n; ++k) {
            // 1. Update 对新加入的点进行延申
            for (Node node : adjacency[newest]) {
                if (!visit[node.to] && distance[node.to] > distance[newest] + node.weight) {
                    distance[node.to] = distance[newest] + node.weight;
                    queue.add(new Node(node.to, distance[node.to]));
                }
            }
            // 2. Scan 找到连接已选择部分和未选择部分的最小边
            Node next = findMinNode(queue, visit);
            // 没有可以继续延申的点了，剩下的是不可达的
            if (next == null) {
                break;
            }
            // 3. Add 加入已选择顶点部分
            newest = next.to;
            visit[newest] = true;
        }
        return distance;
    }

    @Override
    public int[][] getFullSourceShortPath(int[][] matrix, int inf) {
        int n = matrix.length;
        int[][] distance = new int[n][];
        for (int i = 0; i < n; i++) {
            distance[i] = getSingleSourceShortPath(i, matrix, inf);
        }
        return distance;
    }

    @Override
    public int[][] getFullSourceShortPath(List<Node>[] adjacency) {
        int n = adjacency.length;
        int[][] distance = new int[n][];
        for (int i = 0; i < n; i++) {
            distance[i] = getSingleSourceShortPath(i, adjacency);
        }
        return distance;
    }
}
```

#### topological

##### TopologicalSort

```java
/**
 * @author : Cattle_Horse
 * @date : 2023/12/2 16:57
 * @description :拓扑排序接口
 **/
interface TopologicalSort {

    /**
     * 拓扑排序
     *
     * @param result 排序后的拓扑序列存储位置（需满足长度大于等于图的点数）
     * @param matrix 待拓扑排序的邻接矩阵（无权值）
     * @return 是否存在拓扑序列
     */
    boolean applyUnweighted(int[] result, boolean[][] matrix);

    /**
     * 拓扑排序
     *
     * @param result 排序后的拓扑序列存储位置（需满足长度大于等于图的点数）
     * @param adj    待拓扑排序的邻接表（无权值）
     * @return 是否存在拓扑序列
     */
    boolean applyUnweighted(int[] result, List<Integer>[] adj);

    /**
     * 拓扑排序
     *
     * @param result 排序后的拓扑序列存储位置（需满足长度大于等于图的点数）
     * @param matrix 待拓扑排序的邻接矩阵（有权值）
     * @param inf    标志矩阵中不存在边的值
     * @return 是否存在拓扑序列
     */
    boolean apply(int[] result, int[][] matrix, final int inf);

    /**
     * 拓扑排序
     *
     * @param result 排序后的拓扑序列存储位置（需满足长度大于等于图的点数）
     * @param adj    待拓扑排序的邻接表（有权值）
     * @return 是否存在拓扑序列
     */
    boolean apply(int[] result, List<Node>[] adj);
}
```

##### Kahn

```java
/**
 * @author : Cattle_Horse
 * @date : 2023/12/5 20:04
 * @description : Kahn算法实现拓扑排序<BR>
 * 一直选择入度为0的点，并把它删去<BR>
 * {@code apply} 或 {@code applyUnweighted} 方法的时间复杂度均为 {@code O( V+E )}，其中 V 是图中顶点个数
 * @see TopologicalSort
 **/
class Kahn implements TopologicalSort {
    private Deque<Integer> findZeroIndex(int[] arr) {
        int n = arr.length;
        Deque<Integer> deque = new ArrayDeque<>(n);
        for (int i = 0; i < n; ++i) {
            if (arr[i] == 0) {
                deque.addLast(i);
            }
        }
        return deque;
    }

    @Override
    public boolean applyUnweighted(int[] result, boolean[][] matrix) {
        int n = matrix.length;
        int[] inDegree = new int[n];
        for (boolean[] froms : matrix) {
            for (int j = 0; j < n; j++) {
                if (froms[j]) {
                    ++inDegree[j];
                }
            }
        }
        int size = 0;
        for (Deque<Integer> deque = findZeroIndex(inDegree); !deque.isEmpty(); ) {
            int index = deque.pollFirst();
            result[size++] = index;
            // remove index node
            inDegree[index] = -1;
            for (int j = 0; j < n; j++) {
                if (matrix[index][j] && --inDegree[j] == 0) {
                    deque.addLast(j);
                }
            }
        }
        return size == n;
    }

    @Override
    public boolean applyUnweighted(int[] result, List<Integer>[] adj) {
        int n = adj.length;
        int[] inDegree = new int[n];
        for (List<Integer> froms : adj) {
            for (Integer to : froms) {
                ++inDegree[to];
            }
        }
        int size = 0;
        for (Deque<Integer> deque = findZeroIndex(inDegree); !deque.isEmpty(); ) {
            int index = deque.pollFirst();
            result[size++] = index;
            // remove index node
            inDegree[index] = -1;
            for (Integer to : adj[index]) {
                if (--inDegree[to] == 0) {
                    deque.addLast(to);
                }
            }
        }
        return size == n;
    }

    @Override
    public boolean apply(int[] result, int[][] matrix, int inf) {
        int n = matrix.length;
        int[] inDegree = new int[n];
        for (int[] froms : matrix) {
            for (int j = 0; j < n; j++) {
                if (froms[j] != inf) {
                    ++inDegree[j];
                }
            }
        }
        int size = 0;
        for (Deque<Integer> deque = findZeroIndex(inDegree); !deque.isEmpty(); ) {
            int index = deque.pollFirst();
            result[size++] = index;
            // remove index node
            inDegree[index] = -1;
            for (int j = 0; j < n; j++) {
                if (matrix[index][j] != inf && --inDegree[j] == 0) {
                    deque.addLast(j);
                }
            }
        }
        return size == n;
    }

    @Override
    public boolean apply(int[] result, List<Node>[] adj) {
        int n = adj.length;
        int[] inDegree = new int[n];
        for (List<Node> froms : adj) {
            for (Node node : froms) {
                ++inDegree[node.to];
            }
        }
        int size = 0;
        for (Deque<Integer> deque = findZeroIndex(inDegree); !deque.isEmpty(); ) {
            int index = deque.pollFirst();
            result[size++] = index;
            // remove index node
            inDegree[index] = -1;
            for (Node node : adj[index]) {
                if (--inDegree[node.to] == 0) {
                    deque.addLast(node.to);
                }
            }
        }
        return size == n;
    }
}
```

##### DFSTopological

```java
/**
 * @author : Cattle_Horse
 * @date : 2023/12/5 19:40
 * @description : 基于后序遍历的拓扑排序<BR>
 * 优先访问最深依赖的点（后序遍历），最后翻转<BR>
 * {@code apply} 或 {@code applyUnweighted} 方法的时间复杂度均为 {@code O( V+E )}，其中 V 是图中顶点个数
 * @see TopologicalSort
 **/
class DFSTopological implements TopologicalSort {
    private static final int UNVISITED = 1;
    private static final int VISITED = 2;
    private static final int VISITING = 4;

    private int n;
    private int size;
    private int inf;

    private int[] initAndGetStatus(int n) {
        this.n = n;
        this.size = 0;
        int[] status = new int[n];
        Arrays.fill(status, UNVISITED);
        return status;
    }

    private int[] initAndGetStatus(int n, int inf) {
        this.inf = inf;
        return this.initAndGetStatus(n);
    }

    private void reverse(int[] arr) {
        for (int l = 0, r = n - 1, temp; l < r; ++l, --r) {
            temp = arr[l];
            arr[l] = arr[r];
            arr[r] = temp;
        }
    }

    /**
     * @return 是否存在访问到visiting的节点
     */
    private boolean dfsUnweighted(int start, int[] status, boolean[][] mat, int[] result) {
        if (status[start] != UNVISITED) {
            return status[start] == VISITING;
        }
        status[start] = VISITING;
        for (int j = 0; j < n; j++) {
            if (mat[start][j] && dfsUnweighted(j, status, mat, result)) {
                return true;
            }
        }
        status[start] = VISITED;
        result[size++] = start;
        return false;
    }

    @Override
    public boolean applyUnweighted(int[] result, boolean[][] matrix) {
        int[] status = initAndGetStatus(matrix.length);
        for (int i = 0; i < n; i++) {
            if (dfsUnweighted(i, status, matrix, result)) {
                return false;
            }
        }
        reverse(result);
        return true;
    }

    private boolean dfsUnweighted(int start, int[] status, List<Integer>[] adj, int[] result) {
        if (status[start] != UNVISITED) {
            return status[start] == VISITING;
        }
        status[start] = VISITING;
        for (Integer to : adj[start]) {
            if (dfsUnweighted(to, status, adj, result)) {
                return true;
            }
        }
        status[start] = VISITED;
        result[size++] = start;
        return false;
    }

    @Override
    public boolean applyUnweighted(int[] result, List<Integer>[] adj) {
        int[] status = initAndGetStatus(adj.length);
        for (int i = 0; i < n; i++) {
            if (dfsUnweighted(i, status, adj, result)) {
                return false;
            }
        }
        reverse(result);
        return true;
    }

    /**
     * @return 是否存在访问到visiting的节点
     */
    private boolean dfs(int start, int[] status, int[][] mat, int[] result) {
        if (status[start] != UNVISITED) {
            return status[start] == VISITING;
        }
        status[start] = VISITING;
        for (int j = 0; j < n; j++) {
            if (mat[start][j] != inf && dfs(j, status, mat, result)) {
                return true;
            }
        }
        status[start] = VISITED;
        result[size++] = start;
        return false;
    }

    @Override
    public boolean apply(int[] result, int[][] matrix, int inf) {
        int[] status = initAndGetStatus(matrix.length);
        for (int i = 0; i < n; i++) {
            if (dfs(i, status, matrix, result)) {
                return false;
            }
        }
        reverse(result);
        return true;
    }


    private boolean dfs(int start, int[] status, List<Node>[] adj, int[] result) {
        if (status[start] != UNVISITED) {
            return status[start] == VISITING;
        }
        status[start] = VISITING;
        for (Node node : adj[start]) {
            if (dfs(node.to, status, adj, result)) {
                return true;
            }
        }
        status[start] = VISITED;
        result[size++] = start;
        return false;
    }

    @Override
    public boolean apply(int[] result, List<Node>[] adj) {
        int[] status = initAndGetStatus(adj.length);
        for (int i = 0; i < n; i++) {
            if (dfs(i, status, adj, result)) {
                return false;
            }
        }
        reverse(result);
        return true;
    }
}
```
