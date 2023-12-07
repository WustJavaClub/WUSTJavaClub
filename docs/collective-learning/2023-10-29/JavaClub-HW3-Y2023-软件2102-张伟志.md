# 第三次作业报告

姓名：张伟志

班级：软件2102

## 汉诺塔

### 汉诺三塔

**非递归实现汉诺塔并输出搬动过程**

```java
import java.util.Deque;
import java.util.LinkedList;

public class Main {
    public static void main(String[] args) {
        int n = 3;
        char a = 'a', b = 'b', c = 'c';
        new Hanoi().recursion(n, a, b, c);
        System.out.println("------------------");
        new Hanoi().cycle(n, a, b, c);
    }
}

class Hanoi {
    /**
     * 非递归实现汉诺塔
     */
    void cycle(int n, char a, char b, char c) {
        Deque<Item> stack = new LinkedList<>();
        stack.addLast(new Item(n, a, b, c));
        while (!stack.isEmpty()) {
            Item item = stack.pollLast();
            if (item.n == 1) {
                move(item.start, item.target);
            } else {
                stack.addLast(new Item(item.n - 1, item.temp, item.start, item.target));
                stack.addLast(new Item(1, item.start, item.temp, item.target));
                stack.addLast(new Item(item.n - 1, item.start, item.target, item.temp));
            }
        }
    }

    /**
     * 将 n 个盘片借助temp柱子，从start柱子移动到target柱子上
     */
    void recursion(int n, char start, char temp, char target) {
        if (n == 1) {
            move(start, target);
        } else {
            recursion(n - 1, start, target, temp);
            recursion(1, start, temp, target);
            recursion(n - 1, temp, start, target);
        }
    }
    
    void move(char x, char y) {
        System.out.printf("%c -> %c\n", x, y);
    }

    static class Item {
        int n;
        char start, temp, target;

        public Item(int n, char start, char temp, char target) {
            this.n = n;
            this.start = start;
            this.temp = temp;
            this.target = target;
        }
    }
}

```



题目链接：[T291125 Tower of Hanoi - 洛谷](https://www.luogu.com.cn/problem/T291125)
$$
\begin{aligned}
Hanoi(n) &=
	\begin{cases}
 		2 \times Hanoi(n-1) + 1 & \text{ if } n > 0 \\
 		1 & \text{ if } n= 0
	\end{cases}
	\\
Hanoi(n) &= 2^{n}-1
\end{aligned}
$$

**BigInteger**

`BigInteger` 自带幂级数取模

如果 $2^{n} \equiv 0 \pmod{m}$，减一可能出现负数，而 $a + m \equiv a \pmod{m}$，因此加一个 $m$ 防止错误

```java
import java.math.BigInteger;
import java.util.Scanner;

public class T291125 {
    static final BigInteger MOD = BigInteger.valueOf(998244353);
    static final BigInteger ONE = BigInteger.valueOf(1);
    static final BigInteger TWO = BigInteger.valueOf(2);

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        BigInteger n = sc.nextBigInteger();
        System.out.println(TWO.modPow(n, MOD).subtract(ONE).add(MOD).mod(MOD));
    }
}
```

**快速幂**

- “乘后取模”与“边模边乘”结果一样：$(a\times b) \% m = ((a\%m)\times(b\%m))\%m$

- 快速幂：底数平方，指数减半

  如 $2^{32}=(2\times 2)^{16}=(4\times 4)^{8}=(16\times 16)^{4}=(256\times 256)^2=(4294967296)^1$

  对于求 $a^b$ 原本需要循环执行 $b$ 次，现在只需要执行 $\log_2 b$ 次了

  当指数不是偶数时，只需要取出一个底数就能变成偶数了

  如 $2^{9}=2\times 2^8=2\times (2\times 2)^4=2\times (4\times 4)^2=2\times (16\times 16)^1=512$ 

```java
import java.util.Scanner;

public class T291125 {
    /**
     * 递归实现快速幂
     *
     * @return a^b % mod
     */
    long powMod(long a, long b, final long mod) {
        if (b == 0) {
            return 1 % mod;
        }
        if (b == 1) {
            return a % mod;
        }
        if (b % 2 == 1) {
            return a * powMod(a * a % mod, (b - 1) / 2, mod) % mod;
        }
        return powMod(a * a % mod, b / 2, mod) % mod;
    }

    /**
     * 循环实现快速幂
     *
     * @return a^b % mod
     */
    /*
    long powMod(long a, long b, final long mod) {
        long ret = 1;
        while (b != 0) {
            if (b % 2 == 1) {
                ret = ret * a % mod;
            }
            b /= 2;
            a = a * a % mod;
        }
        return ret % mod;
    }
     */

    static final long MOD = 998244353;

    public void main(Scanner sc) {
        long n = sc.nextLong();
        System.out.println((powMod(2, n, MOD) - 1 + MOD) % MOD);
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        new T291125().main(sc);
    }
}
```

### 汉诺四塔

[多塔汉诺塔问题 - 维基百科](https://zh.wikipedia.org/wiki/汉诺塔#多塔汉诺塔问题)

题目链接：[96. 奇怪的汉诺塔](https://www.acwing.com/problem/content/98/)
$$
\begin{aligned}
Hanoi(n,3) &= 2^n - 1\\
Hanoi(n,4) &= 
	\begin{cases}
		\min_{m \in[1,n-1]}(2Hanoi(m,4) + Hanoi(n-m, 3)) & \text{ if } n > 1\\
		1 & \text{ if } n = 1\\
		0 & \text{ if } n = 0
	\end{cases}
\end{aligned}
$$

```java
public class Problem96 {
    static final int N = 12;

    public static void main(String[] args) {
        int[] hanoi = new int[N + 1];
        hanoi[0] = 0;
        hanoi[1] = 1;
        for (int i = 2; i <= N; i++) {
            int min = Integer.MAX_VALUE;
            for (int j = 1; j < i; j++) {
                min = Math.min(min, 2 * hanoi[i - j] + (1 << j) - 1);
            }
            hanoi[i] = min;
        }
        for (int i = 1; i <= N; i++) {
            System.out.println(hanoi[i]);
        }
    }
}
```

## 分平面

题目链接：[T291123 Barmecide Feast](https://www.luogu.com.cn/problem/T291123)
$$
\begin{aligned}
L_{n} &= 
\begin{cases}
	L_{n-1} + n	& \text{ if } n > 0\\
	1 & \text{ if } n = 0
\end{cases}\\
L_n &= \dfrac{(n+1)\cdot n}{2}+1
\end{aligned}
$$

```java
import java.util.Scanner;

public class T291123 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        long n = sc.nextLong();
        System.out.println(n * (1 + n) / 2 + 1);
    }
}
```

## 约瑟夫

### P1996 约瑟夫问题

题目链接： [P1996 约瑟夫问题](https://www.luogu.com.cn/problem/P1996)

报数变量为 $number$，在其加 $1$ 模 $m$ 后值域为 $[0, m-1]$

为了将其映射到 $[1, m]$

- 在判断 $number = m$ 并输出当前值之后将 $number$ 置为 $0$

- 考虑把括号的一个 $1$ 移动到取模之后，$(number+1) \% m \rightarrow number \% m + 1 \% m \rightarrow number \% m +1$

**输入部分**

```java
public class P1996 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt(), m = sc.nextInt();
        new Solution1().main(n, m);
        new Solution2().main(n, m);
    }
}
```

**队列实现**

```java
class Solution1 {
    public void main(int n, int m) {
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 1; i <= n; i++) {
            queue.add(i);
        }
        // 当前报数
        int number = 1;
        while (!queue.isEmpty()) {
            int current = queue.poll();
            if (number != m) {
                queue.add(current);
            } else {
                System.out.print(current + " ");
                // number = 0;
            }
            // ++number;
            number = number % m + 1;
        }
    }
}
```

**循环单链表实现**

```java
class Solution2 {
    public void main(int n, int m) {
        // 虚拟头节点
        Node header = new Node(-1, null);
        addValue(header, n);
        // 报数
        int number = 1;
        // 剩余人数
        int rest = n;
        while (rest != 0) {
            Node current = header.next.next;
            if (number == m) {
                System.out.print(current.value + " ");
                header.next.next = current.next;
                number = 0;
                --rest;
            } else {
                header.next = current;
            }
            ++number;
        }
    }

    /**
     * 插入元素到虚拟头节点的单链表中，并构成环
     * 运行后header虚拟头节点将指向真正头节点前一个
     *
     * @param header 虚拟头节点
     * @param n      插入元素个数
     */
    void addValue(Node header, int n) {
        Node tail = header;
        for (int i = 1; i <= n; i++) {
            tail.next = new Node(i, null);
            tail = tail.next;
        }
        tail.next = header.next;
        header.next = tail;
    }

    static class Node {
        int value;
        Node next;

        public Node(int value, Node next) {
            this.value = value;
            this.next = next;
        }
    }
}
```

### T291920 Countdown to Death

题目链接： [T291920 Countdown to Death](https://www.luogu.com.cn/problem/T291920)
$$
J(n,k)=\begin{cases}
(J(n-1,k)+k-1) \bmod n +1 & \text{ if } n > 1\\
1 & \text{ if } n = 1
\end{cases}
$$

```java
import java.util.Scanner;

public class T291920 {
    void main(Scanner sc) {
        int n = sc.nextInt(), k = sc.nextInt();
        int ans = 1;
        for (int i = 2; i <= n; i++) {
            ans = (ans + k - 1) % i + 1;
        }
        System.out.println(ans);
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        new T291920().main(sc);
    }
}
```

## 容斥原理

题目链接：[U375671 能被整除的数](https://www.luogu.com.cn/problem/U375671)
$$
\begin{aligned}
G(p_1, \cdots ,p_{n-1}, p_{n}) &= \begin{cases}
	G(p_1, \cdots ,p_{n-1}) + p_{n} - G(p_1, \cdots ,p_{n-1}) \cap p_{n} & \text{ if } n > 1\\
	p_1 & \text{ if } n = 1\\
\end{cases}\\
\text{项数}(G(p_1, \cdots ,p_{n-1}, p_{n})) &= 2^{n}-1\\
&\text{奇加偶减}\newline\newline
lcm(a, b)  &= \dfrac{a\cdot b}{\gcd(a, b)}\newline\newline
\gcd(a, b) &= \gcd(b, a\% b)\newline\newline
\end{aligned}
$$

**输入及部分方法**

```java
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class U375671 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt(), m = sc.nextInt();
        int[] arr = new int[m];
        for (int i = 0; i < m; i++) {
            arr[i] = sc.nextInt();
        }
        new Solution1().main(n, m, arr);
        new Solution2().main(n, m, arr);
        new Solution3().main(n, m, arr);
    }

    /**
     * 判断是否是奇数
     */
    static boolean isOdd(int x) {
        return x % 2 == 1;
    }

    /**
     * 两数的最大公约数
     */
    static long gcd(long a, long b) {
        if (b == 0) {
            return a;
        }
        return gcd(b, a % b);
    }

    /**
     * 两数的最小公倍数
     */
    static long lcm(long a, long b) {
        return a / gcd(a, b) * b;
    }

    /**
     * 获得 [1,n] 中，能被 factor 整除的数的个数
     *
     * @param n      右区间
     * @param factor 被整除的因子
     * @return [1, n]中能被 factor 整除的数的个数
     */
    static long getNumber(long n, long factor) {
        return n / factor;
    }
}
```

**递归项**

记录每一项进行递归

```java
class Solution1 {
    int n;

    List<Item> getUnion(int[] arr, int index) {
        List<Item> items = new ArrayList<>((1 << (index + 1)) - 1);
        if (index == 0) {
            items.add(new Item(1, arr[index]));
            return items;
        }
        List<Item> previous = getUnion(arr, index - 1);
        items.addAll(previous);
        items.add(new Item(1, arr[index]));
        for (Item prev : previous) {
            items.add(new Item(
                    prev.cnt + 1,
                    prev.lcm > n ? prev.lcm : U375671.lcm(arr[index], prev.lcm)));
        }
        return items;
    }

    public void main(int n, int m, int[] arr) {
        this.n = n;
        List<Item> items = getUnion(arr, m - 1);
        int ans = 0;
        for (Item item : items) {
            if (U375671.isOdd(item.cnt)) {
                ans += U375671.getNumber(n, item.lcm);
            } else {
                ans -= U375671.getNumber(n, item.lcm);
            }
        }
        System.out.println(ans);
    }

    static class Item {
        int cnt;
        long lcm;

        public Item(int cnt, long lcm) {
            this.cnt = cnt;
            this.lcm = lcm;
        }
    }
}
```

**递归枚举子集**

```java
class Solution2 {
    int n, m;

    /**
     * 递归枚举排列情况，对于每一个index有选和不选两种情况
     *
     * @param index 当前判断arr[index]
     * @param cnt   已经选了的数的个数
     * @param lcm   已经选了的数的最小公倍数
     */
    long dfs(int[] arr, int index, int cnt, long lcm) {
        // 所有index都判断完了
        if (index == m) {
            if (cnt == 0) {
                return 0;
            }
            if (U375671.isOdd(cnt)) {
                return U375671.getNumber(n, lcm);
            }
            return -U375671.getNumber(n, lcm);
        }
        long ans = 0;
        // 选择 arr[index]
        ans += dfs(arr, index + 1, cnt + 1, lcm > n ? lcm : U375671.lcm(lcm, arr[index]));
        // 不选 arr[index]
        ans += dfs(arr, index + 1, cnt, lcm);
        return ans;
    }

    public void main(int n, int m, int[] arr) {
        this.n = n;
        this.m = m;
        System.out.println(dfs(arr, 0, 0, 1));
    }
}
```

**二进制枚举子集**

- `i >> j & 1`：取出 $i$ 的二进制第 $j$ 位的值
- `1 << i`：$2^i$

```java
class Solution3 {
    public void main(int n, int m, int[] arr) {
        long ans = 0;
        int s = 1 << m;
        for (int i = 1; i < s; ++i) {
            int cnt = 0;
            long lcm = 1;
            for (int j = 0; j < m; ++j) {
                if ((i >> j & 1) == 1) {
                    ++cnt;
                    lcm = U375671.lcm(lcm, arr[j]);
                    if (lcm > n) {
                        break;
                    }
                }
            }
            if (U375671.isOdd(cnt)) {
                ans += U375671.getNumber(n, lcm);
            } else {
                ans -= U375671.getNumber(n, lcm);
            }
        }
        System.out.println(ans);
    }
}
```

## 0-1 背包

题目链接：[2. 01背包问题](https://www.acwing.com/problem/content/description/2/)

$$
F(n,v) = \begin{cases}
	\max(F(n - 1, v - volume_i) + value_i,F(n - 1, v)) & \text{ if } v > 0 \: \& \: n > 0 \: \& \: v \ge volume_i\\
	F(n-1,v) & \text{ if } v > 0 \: \& \: n > 0 \: \& \: v < volume_i\\
	0 & \text{ if } v = 0 \: \& \: n > 0\\
	0 & \text{ if } n = 0
\end{cases}
$$
**输入部分**

```java
public class Problem2 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt(), v = sc.nextInt();
        int[] weights = new int[n];
        int[] values = new int[n];
        for (int i = 0; i < n; i++) {
            weights[i] = sc.nextInt();
            values[i] = sc.nextInt();
        }
        new Solution2().main(n, v, weights, values);
    }
}
```

**递归实现**

```java
class Solution1 {
    int[] weights, values;
    int[][] memory;
    static final int INITIALIZATION = -1;

    int dfs(int index, int volume) {
        if (index == -1) {
            return 0;
        }
        if (memory[index][volume] != INITIALIZATION) {
            return memory[index][volume];
        }
        if (volume == 0) {
            memory[index][volume] = 0;
            return 0;
        }
        int ans;
        if (volume >= weights[index]) {
            ans = Math.max(dfs(index - 1, volume - weights[index]) + values[index], dfs(index - 1, volume));
        } else {
            ans = dfs(index - 1, volume);
        }
        memory[index][volume] = ans;
        return ans;
    }

    public void main(int n, int v, int[] weights, int[] values) {
        this.weights = weights;
        this.values = values;
        memory = new int[n][v + 1];
        for (int[] row : memory) {
            Arrays.fill(row, INITIALIZATION);
        }
        System.out.println(dfs(n - 1, v));
    }
}
```

**循环实现**

```java
class Solution2 {
    public void main(int n, int v, int[] weights, int[] values) {
        int[][] dp = new int[n][v + 1];
        // Base Case
        for (int i = 0; i < n; i++) {
            dp[i][0] = 0;
        }
        for (int j = 0; j <= v; j++) {
            dp[0][j] = j >= weights[0] ? values[0] : 0;
        }
        // Recursive Case
        for (int i = 1; i < n; i++) {
            for (int j = 1; j <= v; j++) {
                if (j >= weights[i]) {
                    dp[i][j] = Math.max(dp[i - 1][j - weights[i]] + values[i], dp[i - 1][j]);
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        System.out.println(dp[n - 1][v]);
    }
}
```

## 整数分拆

题目链接：[U375865 整数拆分](https://www.luogu.com.cn/problem/U375865)

$$
p(n,k) = \begin{cases}
	1 & \text{ if } n = 1 \\
	1 & \text{ if } k = 1 \\
	p(n,n) & \text{ if } n < k \\
	p(n,k-1) + 1 & \text{ if } n = k \\
	p(n,k-1) + p(n-k, k) & \text{o.w.}
\end{cases}
$$
**输入部分**

```java
public class U375865 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt(), k = sc.nextInt();
        new Solution1().main(n, k);
        new Solution2().main(n, k);
    }
}
```

**递归实现**

```java
class Solution1 {
    static final int INITIALIZATION = -1;
    static final int MOD = (int) 1e9 + 7;

    int[][] memory;

    /**
     * 整数拆分，将 n 拆分成最大值不超过 k 的数的和，返回其方案数
     *
     * @param n 待拆分数
     * @param k 拆分过程中的最大值
     */
    int integerPartition(int n, int k) {
        if (memory[n][k] != INITIALIZATION) {
            return memory[n][k];
        }
        int ans;
        if (n == 1 || k == 1) {
            return 1;
        }
        if (n < k) {
            ans = integerPartition(n, n) % MOD;
        } else if (n == k) {
            ans = (1 + integerPartition(n, k - 1)) % MOD;
        } else {
            ans = (integerPartition(n - k, k) + integerPartition(n, k - 1)) % MOD;
        }
        memory[n][k] = ans;
        return ans;
    }

    public void main(int n, int k) {
        memory = new int[n + 1][k + 1];
        for (int i = 0; i <= n; ++i) {
            Arrays.fill(memory[i], INITIALIZATION);
        }
        System.out.println(integerPartition(n, k));
    }
}
```

**循环实现**

```java
class Solution2 {
    static final int MOD = (int) 1e9 + 7;

    public void main(int n, int k) {
        int[][] dp = new int[n + 1][k + 1];
        // Base Case
        for (int i = 1; i <= n; i++) {
            dp[i][1] = 1;
        }
        for (int i = 1; i <= k; i++) {
            dp[1][i] = 1;
        }
        // Recursive Case
        for (int i = 2; i <= n; i++) {
            for (int j = 2; j <= k; j++) {
                if (i < j) {
                    dp[i][j] = dp[i][i] % MOD;
                } else if (i == j) {
                    dp[i][j] = (1 + dp[i][j - 1]) % MOD;
                } else {
                    dp[i][j] = (dp[i - j][j] + dp[i][j - 1]) % MOD;
                }
            }
        }
        System.out.println(dp[n][k]);
    }
}
```

## 编辑距离

题目链接：[72. 编辑距离](https://leetcode.cn/problems/edit-distance/)
$$
\begin{aligned}
	F(n,m) &= \begin{cases}
		n & \text{ if } m = 0 \\
		m & \text{ if } n = 0 \\
		\min(op_{insert},op_{delete},op_{replace}) & \text{o.w.}\\
	\end{cases}\\\\
	\text{其中：} op_{insert} &= F(n-1,m) + 1\\
	op_{delete} &= F(n,m-1) + 1\\
	op_{replace} &= F(n-1,m-1) + \delta (word1[m-1] \overset{\text{?}}{=} word2[n-1])
\end{aligned}
$$
**递归实现**

```java
class Solution1 {
    char[] word1, word2;
    int[][] memory;

    int dfs(int n, int m) {
        if (n == -1) {
            return m + 1;
        }
        if (m == -1) {
            return n + 1;
        }
        if (memory[n][m] != -1) {
            return memory[n][m];
        }
        int insert = dfs(n, m - 1) + 1;
        int delete = dfs(n - 1, m) + 1;
        int replace = dfs(n - 1, m - 1) +
                (word1[n] == word2[m] ? 0 : 1);
        int ans = Math.min(insert,
                Math.min(delete, replace));
        memory[n][m] = ans;
        return ans;
    }

    public int minDistance(String word1, String word2) {
        int n = word1.length(), m = word2.length();
        memory = new int[n][m];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(memory[i], -1);
        }
        this.word1 = word1.toCharArray();
        this.word2 = word2.toCharArray();
        return dfs(n - 1, m - 1);
    }
}
```

**循环实现**

```java
class Solution2 {
    public int minDistance(String word1, String word2) {
        int n = word1.length(), m = word2.length();
        int[][] dp = new int[n + 1][m + 1];
        // Base Case
        for (int i = 0; i <= n; i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j <= m; j++) {
            dp[0][j] = j;
        }
        // Recursive Case
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                int insert = dp[i][j - 1] + 1;
                int delete = dp[i - 1][j] + 1;
                int replace = dp[i - 1][j - 1] + (word1.charAt(i - 1) == word2.charAt(j - 1) ? 0 : 1);
                dp[i][j] = Math.min(insert,
                        Math.min(delete, replace));
            }
        }
        return dp[n][m];
    }
}
```

## 峰值查找

### 一维

题目链接：[162. 寻找峰值](https://leetcode.cn/problems/find-peak-element/description/)

```java
public class Problem162 {
    public int findPeakElement(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        // 在 nums[left..right] 中查找峰值
        while (left < right) {
            int mid = (left + right) / 2;

            int leftValue = mid - 1 < left ? Integer.MIN_VALUE : nums[mid - 1];
            int rightValue = mid + 1 > right ? Integer.MIN_VALUE : nums[mid + 1];

            if (nums[mid] < rightValue) {
                // 右侧上升，右侧一定有峰值
                // 下一轮搜索的区间 [mid+1..right]
                left = mid + 1;
            } else if (nums[mid] < leftValue) {
                // 左侧上升，左侧一定有峰值
                // 下一轮搜索的区间 [left..mid-1]
                right = mid - 1;
            } else {
                // 两侧都比当前低，则此时已经是峰值
                return mid;
            }
        }
        return left;
    }
}
```

### 二维

题目链接：[1901. 寻找峰值 II](https://leetcode.cn/problems/find-a-peak-element-ii/)

```java
public class Problem1901 {
    public int[] findPeakGrid(int[][] mat) {
        int rows = mat.length, cols = mat[0].length;
        int left = 0, right = cols - 1;
        while (left <= right) {
            int col = left + right >> 1;
            int row = getMaxValueIndex(mat, col);
            int currentValue = mat[row][col];
            int leftValue = getValue(mat, row, col - 1, 0, left, rows - 1, right);
            int rightValue = getValue(mat, row, col + 1, 0, left, rows - 1, right);
            if (currentValue < leftValue) {
                right = col - 1;
            } else if (currentValue < rightValue) {
                left = col + 1;
            } else {
                return new int[]{row, col};
            }
        }
        return null;
    }

    int getValue(int[][] mat, int row, int col, int minRow, int minCol, int maxRow, int maxCol) {
        if (row < minRow || col < minCol || row > maxRow || col > maxCol) {
            return Integer.MIN_VALUE;
        }
        return mat[row][col];
    }

    int getMaxValueIndex(int[][] mat, int col) {
        int maxValue = Integer.MIN_VALUE, maxValueIndex = -1;
        for (int i = 0; i < mat.length; i++) {
            if (mat[i][col] > maxValue) {
                maxValue = mat[i][col];
                maxValueIndex = i;
            }
        }
        return maxValueIndex;
    }
}
```
