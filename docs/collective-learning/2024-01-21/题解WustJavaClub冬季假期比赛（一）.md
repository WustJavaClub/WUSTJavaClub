## WustJavaClub冬季假期比赛（一）

### 说明

知识点处给出了相对应知识点的力扣练习链接，可以多刷相应题目掌握相关算法

部分题目由于洛谷平台的限制，没有给Java语言足够的空间解决问题，而在蓝桥杯比赛中Java语言会相较于C/C++多出两倍的空间，在比赛时可以不考虑，练习时可以通过 输入优化（[java streamtokenizer](https://www.bing.com/search?q=java+streamtokenizer)） 以及 输出优化（[java PrintWriter](https://www.bing.com/search?q=java+PrintWriter)） 解决问题，输入优化可以参见[最大子段和](###最大子段和)

### 不浪费原料的汉堡制作方案

题目链接：[U399513 不浪费原料的汉堡制作方案 - 洛谷](https://www.luogu.com.cn/problem/U399513)

题目来源：[1276. 不浪费原料的汉堡制作方案 - 力扣](https://leetcode.cn/problems/number-of-burgers-with-no-waste-of-ingredients/)

知识点：[数学](https://leetcode.cn/problemset/?page=1&topicSlugs=math)

题目可以转化为：

求解以下关于 $x,y$ 的二元一次方程组的整数解（其中 $a,b$ 为整数）
$$
\begin{cases}
4x + 2y = a\\
x + y = b
\end{cases}
$$
解得：
$$
\begin{cases}
x = \dfrac{a}{2}-b\\
y = 2b-\dfrac{a}{2}
\end{cases}
$$
由题意，$x,y \ge 0$ 且 $x,y\in \mathbb{N}$，即：
$$
\begin{cases}
a=2k,k\in \mathbb{N}\\
a\ge 2b\\
4b\ge a
\end{cases}
$$
若不满足，则无解。

```java
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        for (int i = 0; i < t; i++) {
            int a = sc.nextInt(), b = sc.nextInt();
            System.out.println(solve(a, b));
        }
    }

    public static String solve(int a, int b) {
        if (a % 2 != 0 || a < 2 * b || b * 4 < a) {
            return "WustJavaClub";
        }
        return (a / 2 - b) + " " + (2 * b - a / 2);
    }
}
```

### 有效的括号

题目链接：[U399370 有效的括号 - 洛谷](https://www.luogu.com.cn/problem/U399370)

题目来源：[20. 有效的括号 - 力扣](https://leetcode.cn/problems/valid-parentheses/description/)

知识点：[栈](https://leetcode.cn/problemset/?page=1&topicSlugs=stack)

具体原理参见[力扣官方题解](https://leetcode.cn/problems/valid-parentheses/solutions/373578/you-xiao-de-gua-hao-by-leetcode-solution/)

> 判断括号的有效性可以使用「栈」这一数据结构来解决。
>
> 我们遍历给定的字符串 $s$。当我们遇到一个左括号时，我们会期望在后续的遍历中，有一个相同类型的右括号将其闭合。由于后遇到的左括号要先闭合，因此我们可以将这个左括号放入栈顶。
>
> 当我们遇到一个右括号时，我们需要将一个相同类型的左括号闭合。此时，我们可以取出栈顶的左括号并判断它们是否是相同类型的括号。如果不是相同的类型，或者栈中并没有左括号，那么字符串 $s$ 无效，返回 `False`。
>
> 在遍历结束后，如果栈中没有左括号，说明我们将字符串 $s$ 中的所有左括号闭合，返回 `True`，否则返回 `False`。
>
> 注意到有效字符串的长度一定为偶数，因此如果字符串的长度为奇数，我们可以直接返回 `False`，省去后续的遍历判断过程。

```java
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Scanner;

public class Main {
    /**
     * 获取与右括号匹配的左括号
     */
    private static char getMatch(char c) {
        if (c == ')') {
            return '(';
        }
        if (c == '}') {
            return '{';
        }
        return '[';
    }

    private static boolean isValid(String s) {
        int n = s.length();
        if (n % 2 == 1) {
            return false;
        }
        Deque<Character> stack = new ArrayDeque<>(n);
        for (int i = 0; i < n; ++i) {
            char c = s.charAt(i);
            if (c == '(' || c == '{' || c == '[') {
                stack.addLast(c);
            } else {
                if (stack.isEmpty() || stack.pollLast() != getMatch(c)) {
                    return false;
                }
            }
        }
        return stack.isEmpty();
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        for (int j = 0; j < n; j++) {
            if (isValid(sc.next())) {
                System.out.println("Yes");
            } else {
                System.out.println("No");
            }
        }
    }
}
```

### 最大和的路径 Ⅱ

题目链接：[U265628 067. 最大和的路径 Ⅱ - 洛谷](https://www.luogu.com.cn/problem/U265628)

题目来源：2021级WustJavaClub第七周作业、[120. 三角形最小路径和 - 力扣](https://leetcode.cn/problems/triangle/description/)

知识点：[记忆化搜索](https://leetcode.cn/problemset/?page=1&topicSlugs=memoization)、[动态规划](https://leetcode.cn/problemset/?page=1&topicSlugs=dynamic-programming)

题解参考[力扣官方题解](https://leetcode.cn/problems/triangle/solutions/329143/san-jiao-xing-zui-xiao-lu-jing-he-by-leetcode-solu/)（最小路径和更改为最大路径和）

```java
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int[][] triangle = new int[n][];
        for (int i = 0; i < n; ++i) {
            triangle[i] = new int[i + 1];
            for (int j = 0; j <= i; ++j) {
                triangle[i][j] = sc.nextInt();
            }
        }
        int[] dp = new int[n];
        for (int i = 0; i < n; ++i) {
            dp[i] = triangle[n - 1][i];
        }
        // 从下向上走
        for (int i = n - 2; i >= 0; --i) {
            for (int j = 0; j <= i; ++j) {
                dp[j] = Math.max(dp[j], dp[j + 1]) + triangle[i][j];
            }
        }
        System.out.println(dp[0]);
    }
}
```

### Nim 游戏

题目链接：[U399391 Nim 游戏 - 洛谷](https://www.luogu.com.cn/problem/U399391)

题目来源：[292. Nim 游戏 - 力扣](https://leetcode.cn/problems/nim-game/description/)

知识点：[脑筋急转弯](https://leetcode.cn/problemset/?page=1&topicSlugs=brainteaser)

- 若剩余 1-3 块石头，则胜，自己可以一次拿完
- 若剩余  4  块石头，则败，无论自己拿多少，在对手阶段会剩下 1-3 块石头
- 若剩余 5-7 块石头，则胜，对于5,6,7块石头，自己拿1,2,3，可以保证在最后一次自己的回合时剩余 1-3 块石头
- 若剩余  8  块石头，则败，自己可以选择 3,2,1，对应的对手可以选择 1,2,3，使得下一次到自己时剩余4块石头
- 以此类推

```Java
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        for (int i = 0; i < n; ++i) {
            if (sc.nextInt() % 4 != 0) {
                System.out.println("Yes");
            } else {
                System.out.println("No");
            }
        }
    }
}
```

### 选数

题目链接：[P1036 [NOIP2002 普及组] 选数](https://www.luogu.com.cn/problem/P1036)

知识点：[搜索](https://leetcode.cn/problemset/?page=1&topicSlugs=depth-first-search) / [二进制枚举](https://leetcode.cn/problemset/?page=1&topicSlugs=bit-manipulation)

对于每一个数有选和不选两种情况，可以通过递归列出全部情况，也可以通过二进制枚举列出全部情况

**递归**

```java
import java.util.Scanner;

public class Main {
    private static int n, k;
    private static int[] a;

    private static boolean isPrime(int n) {
        if (n == 1) {
            return false;
        }
        for (int i = 2, sq = (int) Math.sqrt(n); i <= sq; ++i) {
            if (n % i == 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * 递归下标为index的数选和不选的两种情况
     *
     * @param index 当前要判断下标为index的数
     * @param cnt   目前已经选择了的数的个数
     * @param sum   选择了的数的和
     * @return 满足条件的方案数
     */
    private static int dfs(int index, int cnt, int sum) {
        if (index == n) {
            if (cnt == k && isPrime(sum)) {
                return 1;
            }
            return 0;
        }
        int ans = 0;
        // 选择下标为index的数
        ans += dfs(index + 1, cnt + 1, sum + a[index]);
        // 不选择下标为index的数
        ans += dfs(index + 1, cnt, sum);
        return ans;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        n = sc.nextInt();
        k = sc.nextInt();
        a = new int[n];
        for (int i = 0; i < n; ++i) {
            a[i] = sc.nextInt();
        }
        System.out.println(dfs(0, 0, 0));
    }
}
```

**二进制枚举**

使用二进制 01 串表示每个数选和不选

```java
import java.util.Scanner;

public class Main {
    private static boolean isPrime(int n) {
        if (n == 1) {
            return false;
        }
        for (int i = 2, sq = (int) Math.sqrt(n); i <= sq; ++i) {
            if (n % i == 0) {
                return false;
            }
        }
        return true;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt(), k = sc.nextInt();
        int[] a = new int[n];
        for (int i = 0; i < n; ++i) {
            a[i] = sc.nextInt();
        }
        int m = 1 << n;
        int ans = 0;
        for (int i = 0; i < m; i++) {
            // 如果二进制中1的个数满足k个，即选了k个数
            if (Integer.bitCount(i) != k) {
                continue;
            }
            // 对选了的数求和
            int sum = 0;
            for (int j = 0; j < n; ++j) {
                // 若对应二进制位为1则表示选了该数
                if ((i >> j & 1) == 1) {
                    sum += a[j];
                }
            }
            if (isPrime(sum)) {
                ans++;
            }
        }
        System.out.println(ans);
    }
}
```

此外，二进制枚举还可以使用 [Gosper's Hack](https://programmingforinsomniacs.blogspot.com/2018/03/gospers-hack-explained.html) 优化，参见[力扣题解方法二](https://leetcode.cn/problems/maximum-rows-covered-by-columns/solutions/2587986/bei-lie-fu-gai-de-zui-duo-xing-shu-by-le-5kb9/?envType=daily-question&envId=2024-01-04)

```Java
public static void main(String[] args) {
    Scanner sc = new Scanner(System.in);
    int n = sc.nextInt(), k = sc.nextInt();
    int[] a = new int[n];
    for (int i = 0; i < n; ++i) {
        a[i] = sc.nextInt();
    }
    int m = 1 << n;
    int ans = 0;
    for (int i = (1 << k) - 1; i < m; ) {
        // 对选了的数求和
        int sum = 0;
        for (int j = 0; j < n; ++j) {
            // 若对应二进制位为1则表示选了该数
            if ((i >> j & 1) == 1) {
                sum += a[j];
            }
        }
        if (isPrime(sum)) {
            ans++;
        }
        int lowbit = i & -i;
        int r = i + lowbit;
        i = ((i ^ r) >> (Integer.numberOfTrailingZeros(i) + 2)) | r;
    }
    System.out.println(ans);
}
```



### 最大子段和

题目链接：[P1115 最大子段和 - 洛谷](https://www.luogu.com.cn/problem/P1115)、[53. 最大子数组和 - 力扣](https://leetcode.cn/problems/maximum-subarray/)

知识点：[动态规划](https://leetcode.cn/problemset/?page=1&topicSlugs=dynamic-programming)

若以 $f(i)$ 表示第 $i$ 结尾的最大子数组和，则对于以第 $i+1$ 个数结尾的最大子数组和可以分为以下两种情况：

1. 在 $f(i)$ 的结尾添加第 $i+1$ 个数
2. 第 $i+1$ 个数单独构成

即 $f(i+1)=\min(f(i)+a_{i+1},a_{i+1})$

以下写法如果不使用优化读入在洛谷上会内存溢出，在蓝桥杯比赛时会给Java选手相较于C/C++选手多2倍空间，这里可以不用管

```java
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StreamTokenizer;

/**
 * @author : Cattle_Horse
 * @date : 2024/1/21 21:17
 * @description : <a href="https://www.luogu.com.cn/problem/P1115">P1115 最大子段和</a>
 **/
public class P1115 {
    public static void main(String[] args) throws IOException {
        StreamTokenizer in = new StreamTokenizer(new BufferedReader(new InputStreamReader(System.in)));
        in.nextToken();
        int n = (int) in.nval;
        int[] a = new int[n];
        for (int i = 0; i < n; ++i) {
            in.nextToken();
            a[i] = (int) in.nval;
        }
        // dp[i]: 以i结尾的最大子数组和
        int[] dp = new int[n];
        dp[0] = a[0];
        int max = dp[0];
        for (int i = 1; i < n; i++) {
            dp[i] = Math.max(dp[i - 1] + a[i], a[i]);
            max = Math.max(max, dp[i]);
        }
        System.out.println(max);
    }
}
```

### 数的划分

题目链接：[P1025 [NOIP2001 提高组] 数的划分](https://www.luogu.com.cn/problem/P1025)

知识点：[搜索](https://leetcode.cn/problemset/?page=1&topicSlugs=depth-first-search)

由于 `1,1,5`、`1,5,1` 和 `5,1,1` 三种分法相同，考虑选择一种方式去重

可以令数字递增或递减保证不会重复

```java
import java.util.Scanner;

public class P1025 {
    /**
     * 将n分为k份，且每一份必须大于等于before
     * @return 满足条件的方案数
     */
    int dfs(int n, int k, int before) {
        if (k == 1) {
            if (n >= before) {
                return 1;
            }
            return 0;
        }
        int ans = 0;
        for (int i = before; i <= n; i++) {
            ans += dfs(n - i, k - 1, i);
        }
        return ans;
    }

    void main(Scanner sc) {
        int n = sc.nextInt(), k = sc.nextInt();
        System.out.println(dfs(n, k, 1));
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        new P1025().main(sc);
    }
}
```

### 发射站

题目链接：[P1901 发射站 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P1901)

知识点：[单调栈](https://leetcode.cn/problemset/?topicSlugs=monotonic-stack)

使用单调栈找到对于 $i$ 的最近的两个更大高度的下标，这两个下标可以接收到 $i$ 发出的信号

同 [P1115 最大子段和 - 洛谷](https://www.luogu.com.cn/problem/P1115) 一样，不使用读入优化会内存溢出，比赛时可以不考虑

```java
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StreamTokenizer;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;

/**
 * @author : Cattle_Horse
 * @date : 2024/1/21 22:23
 * @description : <a href="https://www.luogu.com.cn/problem/P1901">P1901 发射站</a>
 **/
public class P1901 {
    public static void main(String[] args) throws IOException {
        StreamTokenizer in = new StreamTokenizer(new BufferedReader(new InputStreamReader(System.in)));
        in.nextToken();
        int n = (int) in.nval;
        int[] h = new int[n], v = new int[n];
        for (int i = 0; i < n; ++i) {
            in.nextToken();
            h[i] = (int) in.nval;
            in.nextToken();
            v[i] = (int) in.nval;
        }
        int[] receive = new int[n];

        Deque<Integer> stack = new ArrayDeque<>();
        // 找到下一个更大值
        for (int i = 0; i < n; ++i) {
            while (!stack.isEmpty() && h[stack.peekLast()] < h[i]) {
                // i 是栈顶元素的下一个更大值
                // 即 i 可以接收到栈顶元素发出的信号
                int index = stack.pollLast();
                receive[i] += v[index];
            }
            stack.addLast(i);
        }
        stack.clear();
        // 找到上一个更大值
        for (int i = n - 1; i >= 0; --i) {
            while (!stack.isEmpty() && h[stack.peekLast()] < h[i]) {
                // i 是栈顶元素的上一个更大值
                // 即 i 可以接收到栈顶元素发出的信号
                int index = stack.pollLast();
                receive[i] += v[index];
            }
            stack.addLast(i);
        }
        int max = Arrays.stream(receive).max().getAsInt();
        System.out.println(max);
    }
}
```

### 二进制问题

题目链接：[P8764 [蓝桥杯 2021 国 BC] 二进制问题](https://www.luogu.com.cn/problem/P8764)

知识点：[数位DP](https://leetcode.cn/problems/count-the-number-of-powerful-integers/solutions/2595149/shu-wei-dp-shang-xia-jie-mo-ban-fu-ti-da-h6ci/) / [记忆化搜索](https://leetcode.cn/problemset/?page=1&topicSlugs=memoization)

数位DP是一种典型的记忆化搜索套路题目

学习链接：

- [数位DP - Cattle_Horse](https://www.cnblogs.com/Cattle-Horse/p/17086813.html)
- [数位DP - 灵茶山艾府](https://leetcode.cn/problems/count-the-number-of-powerful-integers/solutions/2595149/shu-wei-dp-shang-xia-jie-mo-ban-fu-ti-da-h6ci/)

```java
import java.util.Arrays;
import java.util.Scanner;

public class Main {
    static long n;
    static int k, cnt = 0;
    static int[] LIMIT = new int[64];
    static long[][] dp;

    // [i,cnt) 位出现了 j 个 1, 后面有 dp[i][j] 中选择
    static long dfs(int now, int number1, boolean limit) {
        if (number1 > k) return 0;
        if (now == -1) return number1 == k ? 1 : 0;
        if (!limit && dp[now][number1] != -1) return dp[now][number1];
        // 选 0 的情况
        long ans = dfs(now - 1, number1, limit && LIMIT[now] == 0);
        // 选 1 的情况
        if (!limit || LIMIT[now] == 1) ans += dfs(now - 1, number1 + 1, limit && LIMIT[now] == 1);
        if (!limit) dp[now][number1] = ans;
        return ans;
    }


    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        n = sc.nextLong();
        k = sc.nextInt();
        for (long t = n; t != 0; t >>= 1) LIMIT[cnt++] = (int) (t & 1);
        dp = new long[cnt][k + 1];
        for (int i = 0; i < cnt; ++i) Arrays.fill(dp[i], -1);
        // 从最高位开始选
        System.out.println(dfs(cnt - 1, 0, true));
    }
}
```

