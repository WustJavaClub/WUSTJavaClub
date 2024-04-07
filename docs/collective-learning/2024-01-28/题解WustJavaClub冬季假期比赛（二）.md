## WustJavaClub冬季假期比赛（二）

### 拿出最少数目的魔法豆

题目链接：[U401343 拿出最少数目的魔法豆 - 洛谷](https://www.luogu.com.cn/problem/U401343)

题目来源：[2171. 拿出最少数目的魔法豆 - 力扣](https://leetcode.cn/problems/removing-minimum-number-of-magic-beans/)

知识点：贪心、二分、前缀和

设最后需要使得非空袋子中的豆子数量为 $x$（$0 \le x \le max(beans)$），则：

- 魔法豆小于 $x$ 的袋子需要全部取出
- 其余袋子需要取出至 $x$

可以通过 排序+二分 快速的求出哪些袋子的魔法豆数小于 $x$

即对于排序后的 $beans$，若 $[0,index]$ 为魔法豆数小于 $x$ 的袋子， $[index+1,size)$ 为魔法豆数大于等于 $x$ 的袋子

则此时需要拿出的豆子数为 $\sum_{i=0}^{index}beans[i]+\sum_{i=index+1}^{size-1}(beans[i]-x)$

式子中的两段区间和可用前缀和求出

```java
import java.util.Arrays;
import java.util.Scanner;

public class U401343 {
    /**
     * 找到第一个豆子数大于等于 x 的袋子
     */
    private int lowerBound(int[] beans, int x) {
        int l = 0, r = beans.length;
        while (l < r) {
            int mid = l + r >> 1;
            if (beans[mid] >= x) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }

    public long minimumRemoval(int[] beans) {
        Arrays.sort(beans);
        int n = beans.length;
        long[] prefixSum = new long[n + 1];
        for (int i = 1; i <= n; i++) {
            prefixSum[i] = prefixSum[i - 1] + beans[i - 1];
        }
        long ans = Long.MAX_VALUE;
        for (int x = beans[n - 1]; x >= 0; x--) {
            int index = lowerBound(beans, x);
            // [0, index) 的袋子都需要全部拿完
            // [index, n) 的袋子都需要拿至 x
            ans = Math.min(ans, prefixSum[index] + prefixSum[n] - prefixSum[index] - (long) (n - index) * x);
        }
        return ans;
    }

    void main(Scanner sc) {
        int n = sc.nextInt();
        int[] beans = new int[n];
        for (int i = 0; i < n; i++) {
            beans[i] = sc.nextInt();
        }
        System.out.println(minimumRemoval(beans));
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        new U401343().main(sc);
    }
}
```

式子还可以化简
$$
\begin{align}
\sum_{i=0}^{index}beans[i]+\sum_{i=index+1}^{size-1}(beans[i]-x) &= \sum_{i=0}^{index}beans[i]+\sum_{i=index+1}^{size-1}beans[i]-(size-index-1)\times x\\
&=\sum_{i=0}^{size-1}beans[i]-(size-index-1)\times x
\end{align}
$$

即只需要求 $beans$ 的和

```java
import java.util.Arrays;
import java.util.Scanner;

public class U401343 {
    /**
     * 找到第一个豆子数大于等于 x 的袋子
     */
    private int lowerBound(int[] beans, int x) {
        int l = 0, r = beans.length;
        while (l < r) {
            int mid = l + r >> 1;
            if (beans[mid] >= x) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }
    public long minimumRemoval(int[] beans) {
        Arrays.sort(beans);
        int n = beans.length;
        long sum = 0;
        for (int bean : beans) {
            sum += bean;
        }
        long ans = Long.MAX_VALUE;
        for (int x = beans[n - 1]; x >= 0; x--) {
            int index = lowerBound(beans, x);
            ans = Math.min(ans, sum - (long) (n - index) * x);
        }
        return ans;
    }

    void main(Scanner sc) {
        int n = sc.nextInt();
        int[] beans = new int[n];
        for (int i = 0; i < n; i++) {
            beans[i] = sc.nextInt();
        }
        System.out.println(minimumRemoval(beans));
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        new U401343().main(sc);
    }
}
```

更好的做法参见[力扣官方题解](https://leetcode.cn/problems/removing-minimum-number-of-magic-beans/solutions/1270306/na-chu-zui-shao-shu-mu-de-mo-fa-dou-by-l-dhsl/)

### 白鼠试毒

题目链接：[U288400 白鼠试毒 - 洛谷](https://www.luogu.com.cn/problem/U288400)

题目来源：计算机组成原理-海明码

知识点：二分查找、二进制

任意一个数可以有二进制表示，且这个二进制是唯一的

将瓶子的编号转化为二进制，那么每一个编号对应的二进制是唯一的

设需要 $x$ 个小白鼠，这 $x$ 个小白鼠最多能表示的状态是 $2^x$ 种状态

为了需要的小白鼠数量最少，就要尽可能的利用这些状态

> 若以 $8$ 瓶水举例，将这 $8$ 瓶水按照 $0\sim7$ 标号，对应二进制为
>
> ```
> 水     老鼠一 老鼠二 老鼠三
> 0   =   0     0     0
> 1   =   0     0     1
> 2   =   0     1     0
> 3   =   0     1     1
> 4   =   1     0     0
> 5   =   1     0     1
> 6   =   1     1     0
> 7   =   1     1     1
> ```
>
> 第 $x$ 只小白鼠喝其二进制位第 $x$ 列中 值为 $1$ 的水。
>
> 共需要 $3$ 只小白鼠，则
>
> - 第 $1$ 只小白鼠喝标号为 $4,5,6,7$ 的水
> - 第 $2$ 只小白鼠喝标号为 $2,3,6,7$ 的水
> - 第 $3$ 只小白鼠喝标号为 $1,3,5,7$ 的水

题目变为：寻找最小的正整数 $x$，使得 $2^x\ge n$

这个过程可通过二分查找解决

左边界很容易确定为 $1$，右边界需要通过手动求出大致的 $2$ 的 $x$ 次幂满足其十进制 下 位数大于 $10^{100000}$ 的 $x$

``` java
// System.out.println(BigInteger.TWO.pow(350000).toString().length()); // 105361 > 100000
```

代码如下：

```java
import java.math.BigInteger;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        // System.out.println(BigInteger.TWO.pow(350000).toString().length()); // 105361 > 100000
        Scanner sc = new Scanner(System.in);
        BigInteger n = sc.nextBigInteger();
        int l = 1, r = 350000;
        BigInteger two = BigInteger.valueOf(2);
        while (l < r) {
            int mid = (l + r) / 2;
            if (two.pow(mid).compareTo(n) >= 0) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        System.out.println(l);
    }
}
```

求 $2^x\ge n$，即求 $x \ge \log_2n$，而 $\log_2n$ 就是 $n$ 二进制位数

```java
import java.math.BigInteger;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        BigInteger n = sc.nextBigInteger();
        System.out.println(n.bitLength());
    }
}
```



### Eight Queens Puzzle

题目链接：[U401328 Eight Queens Puzzle - 洛谷](https://www.luogu.com.cn/problem/U401328)

题目来源：[P1219 [USACO1.5] 八皇后 Checker Challenge - 洛谷](https://www.luogu.com.cn/problem/P1219)

知识点：搜索

```java
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class Main {
    //map[i][j]表示i行j列是否已经有皇后了
    static boolean[][] map;
    static int[] ans;
    //记录有多少个解
    static int count = 0, n;

    //判断row行col列是否能放皇后
    static boolean check(int row, int col) {
        //判断(row,col)的 ↑ 上方是否有皇后
        for (int i = 1; i <= row; ++i) {
            if (map[i][col]) {
                return false;
            }
        }
        //判断(row,col)的 ↖ 对角线是否有皇后
        for (int i = row - 1, j = col - 1; i >= 1 && j >= 1; --i, --j) {
            //如果有皇后，则说明当前位置不能放皇后
            if (map[i][j]) {
                return false;
            }
        }
        //判断(row,col) ↗ 的对角线是否有皇后
        for (int i = row - 1, j = col + 1; i >= 1 && j <= n; --i, ++j) {
            //如果有皇后，则说明当前位置不能放皇后
            if (map[i][j]) {
                return false;
            }
        }
        //如果 ↖ ↑ ↗ 三个方向
        return true;
    }

    //寻找第row行的皇后
    static void dfs(int row) {
        //如果已经超出棋盘，则说明找到了一个可行解
        if (row > n) {
            ++count;
            //如果是前三个解，则输出
            if (count <= 3) {
                for (int i = 1; i <= n; ++i) System.out.print(ans[i] + " ");
                System.out.println();
            }
            return;
        }
        //判断row行i列能不能放皇后
        for (int i = 1; i <= n; ++i) {
            //如果(row,j)能放皇后，则继续向下搜索
            if (check(row, i)) {
                ans[row] = i;
                //标记这个地方放了皇后
                map[row][i] = true;
                //搜索下一行
                dfs(row + 1);
                //回溯，撤回标记
                map[row][i] = false;
            }
        }
    }

    public static void main(String[] args)throws Exception {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        n = Integer.parseInt(in.readLine());
        map = new boolean[n + 1][n + 1];
        ans = new int[n + 1];
        dfs(1);
        System.out.println(count);
    }
}
```
原理：

- 一个位置的主对角线`↙↗`的横纵坐标之和相同

- 一个位置的次对角线`↖↘`的横纵坐标之差相同

> 设一个点的坐标为 $(x,y)$
>
> 则其主对角线的坐标可表示为 $(x-t,y+t)$
>
> 副对角线的坐标可表示为 $(x+t,y+t)$

根据这个性质，可以使用一个一维数组来标记某一个对角线是否存在皇后

竖直向上的位置同样可以用一个一维数组来标记

```java
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class Main {
    static boolean[] upperLeft;
    static boolean[] upperRight;
    static boolean[] top;
    static int[] ans;
    //记录有多少个解
    static int count = 0, n;

    //判断row行col列是否能放皇后
    static boolean check(int row, int col) {
        if (top[col]) return false;
        if (upperLeft[row - col + n]) return false;
        if (upperRight[row + col]) return false;
        return true;
    }

    //寻找第row行的皇后
    static void dfs(int row) {
        //如果已经超出棋盘，则说明找到了一个可行解
        if (row > n) {
            ++count;
            //如果是前三个解，则输出
            if (count <= 3) {
                for (int i = 1; i <= n; ++i) System.out.print(ans[i] + " ");
                System.out.println();
            }
            return;
        }
        //判断row行i列能不能放皇后
        for (int i = 1; i <= n; ++i) {
            //如果(row,j)能放皇后，则继续向下搜索
            if (check(row, i)) {
                ans[row] = i;
                //标记这个地方放了皇后
                top[i] = true;
                upperLeft[row - i + n] = true;
                upperRight[row + i] = true;
                //搜索下一行
                dfs(row + 1);
                //回溯，撤回标记
                top[i] = false;
                upperLeft[row - i + n] = false;
                upperRight[row + i] = false;
            }
        }
    }

    public static void main(String[] args) throws Exception {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        n = Integer.parseInt(in.readLine());
        top = new boolean[n + 1];
        // 横纵坐标之差的范围在(-n,n)
        // 由于数组无负数下标，给其加上一个偏移量保证它一定在整数范围内
        upperLeft = new boolean[2 * n + 1];
        //横纵坐标之和的范围在[2,2n]
        upperRight = new boolean[2 * n + 1];
        ans = new int[n + 1];
        dfs(1);
        System.out.println(count);
    }
}
```

### 灯泡开关

题目链接：[U401397 灯泡开关 - 洛谷](https://www.luogu.com.cn/problem/U401397)

题目来源：[319. 灯泡开关 - 力扣](https://leetcode.cn/problems/bulb-switcher/description/)

知识点：数学

参见[力扣官方题解](https://leetcode.cn/problems/bulb-switcher/solutions/1099002/deng-pao-kai-guan-by-leetcode-solution-rrgp/)

```java
import java.util.Scanner;

public class Main {
    void main(Scanner sc) {
        int t = sc.nextInt();
        for (int i = 0; i < t; i++) {
            System.out.println((int) Math.sqrt(sc.nextInt()));
        }
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        new Main().main(sc);
    }
}
```



### 数的计算

题目链接：[U289834 数的计算 - 洛谷](https://www.luogu.com.cn/problem/U289834)

题目来源：[P1028 [NOIP2001 普及组] 数的计算 - 洛谷](https://www.luogu.com.cn/problem/P1028)

知识点：

- 递归+记忆化剪枝
- 动态规划

> 递归转化为递推时，需要注意 当依赖某个状态时，那个状态应该已经被更新（或者说是正确的值）

根据题目描述写出递归搜索的代码

```java
import java.util.Arrays;
import java.util.Scanner;

public class Main {

    // now 为当前选择的数
    static int dfs(int now) {
        // 递归出口
        if (now == 1) {
            return 1;
        }
        // 算上自身的一种方案
        int ans = 1;
        for (int i = now / 2; i >= 1; --i) {
            // 这一位选 i 并向下搜索
            ans += dfs(i);
        }
        return ans;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        System.out.println(dfs(n));
    }
}
```

和 $Fibonacci$ 数列一样，搜索子树有大量的重复

将所计算过的状态记录下来，当下次访问时，如果该状态已经计算过，就直接返回

```java
import java.util.Arrays;
import java.util.Scanner;

public class Main {
    static int[] dp;

    static int dfs(int now) {
        // 如果已经计算过该状态了
        if (dp[now] != -1) {
            return dp[now];
        }
        // 递归出口
        if (now == 1) {
            return 1;
        }
        // 算上自身的一种方案
        int ans = 1;
        for (int i = now / 2; i >= 1; --i) {
            // 这一位选 i 并向下搜索
            ans += dfs(i);
        }
        // 先赋值给记忆化数组
        dp[now] = ans;
        return ans;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        dp = new int[n + 1];
        // 给记忆化数组初始化一个不可能的值
        Arrays.fill(dp, -1);
        System.out.println(dfs(n));
    }
}
```

动态规划：

```java
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; ++i) {
            for (int j = i / 2; j >= 1; --j) {
                dp[i] += dp[j]; // 累加这一位选j的总方案
            }
            ++dp[i]; // 算上自身的一种方案
        }
        System.out.println(dp[n]);
    }
}
```

### 一和零

题目链接：[U401382 一和零 - 洛谷](https://www.luogu.com.cn/problem/U401382)

题目来源：[474. 一和零 - 力扣](https://leetcode.cn/problems/ones-and-zeroes/description/)

知识点：动态规划

这题是01背包的变式，即两个容量的01背包

具体解释参见[力扣官方题解](https://leetcode.cn/problems/ones-and-zeroes/solutions/814806/yi-he-ling-by-leetcode-solution-u2z2/)

```java
import java.util.Scanner;

public class Main {
    /**
     * 二维01背包
     */
    public int findMaxForm(String[] strs, int m, int n) {
        int len = strs.length;
        // counts[0][i] : strs[i]中0的个数
        // counts[1][i] : strs[i]中1的个数
        int[][] counts = new int[2][len];
        for (int i = 0; i < len; ++i) {
            for (char c : strs[i].toCharArray()) {
                counts[1][i] += c - '0';
            }
            counts[0][i] = strs[i].length() - counts[1][i];
        }
        // 将 (m,n) 拆分出最多个数 的 和
        // dp[i][j] : 可以放i个0和j个1的最大子集长度
        int[][] dp = new int[m + 1][n + 1];
        for (int i = counts[0][0]; i <= m; ++i) {
            for (int j = counts[1][0]; j <= n; ++j) {
                dp[i][j] = 1;
            }
        }
        for (int k = 1; k < len; ++k) {
            for (int i = m; i >= counts[0][k]; --i) {
                for (int j = n; j >= counts[1][k]; --j) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - counts[0][k]][j - counts[1][k]] + 1);
                }
            }
        }
        return dp[m][n];
    }
    
    void main(Scanner sc) {
        int k = sc.nextInt(), m = sc.nextInt(), n = sc.nextInt();
        String[] strs = new String[k];
        for (int i = 0; i < k; i++) {
            strs[i] = sc.next();
        }
        System.out.println(findMaxForm(strs, m, n));
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        new Main().main(sc);
    }
}
```

### 下一个更大元素 IV

题目链接：[U401409 下一个更大元素 IV - 洛谷](https://www.luogu.com.cn/problem/U401409)

题目来源：[2454. 下一个更大元素 IV - 力扣](https://leetcode.cn/problems/next-greater-element-iv/description/)

知识点：单调栈

题解参见[力扣官方题解](https://leetcode.cn/problems/next-greater-element-iv/solutions/2562064/xia-yi-ge-geng-da-yuan-su-iv-by-leetcode-hjqv/)

```java
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Scanner;

public class Main {
    void main(Scanner sc, PrintWriter out) {
        int n = sc.nextInt();
        int[] nums = new int[n];
        for (int i = 0; i < n; i++) {
            nums[i] = sc.nextInt();
        }
        int[] ans = secondGreaterElement(nums);
        for (int v : ans) {
            out.print(v + " ");
        }
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        PrintWriter out = new PrintWriter(System.out);
        new Main().main(sc, out);
        out.close();
    }

    private static final int NONE = -1;

    public int[] secondGreaterElement(int[] nums) {
        int n = nums.length;
        int[] ans = new int[n];
        Arrays.fill(ans, NONE);
        int[] stack1 = new int[n], stack2 = new int[n];
        int top1 = -1, top2 = -1;
        for (int i = 0; i < n; ++i) {
            while (top2 != -1 && nums[stack2[top2]] < nums[i]) {
                ans[stack2[top2--]] = nums[i];
            }
            int pos = top1;
            while (pos != -1 && nums[stack1[pos]] < nums[i]) {
                --pos;
            }
            for (int j = pos + 1; j <= top1; ++j) {
                stack2[++top2] = stack1[j];
            }
            top1 = pos;
            stack1[++top1] = i;
        }
        return ans;
    }
}
```

### 数字 1 的个数

题目链接：[U401372 数字 1 的个数 - 洛谷](https://www.luogu.com.cn/problem/U401372)

题目来源：[233. 数字 1 的个数 - 力扣](https://leetcode.cn/problems/number-of-digit-one/description/)

知识点：[数位DP](https://leetcode.cn/problems/count-the-number-of-powerful-integers/solutions/2595149/shu-wei-dp-shang-xia-jie-mo-ban-fu-ti-da-h6ci/) / [记忆化搜索](https://leetcode.cn/problemset/?page=1&topicSlugs=memoization)

数位DP是一种典型的记忆化搜索套路题目

同 WustJavaClub冬季假期比赛（一） 这里仅给出参考学习链接：

- [数位DP - Cattle_Horse](https://www.cnblogs.com/Cattle-Horse/p/17086813.html)
- [数位DP - 灵茶山艾府](https://leetcode.cn/problems/count-the-number-of-powerful-integers/solutions/2595149/shu-wei-dp-shang-xia-jie-mo-ban-fu-ti-da-h6ci/)

```java
import java.util.Arrays;
import java.util.Scanner;

public class Main {
    char[] num;
    /**
     * dp[i]  :代表在没有限制的情况下，第i位开始到最低位有多少个1
     * cnt[i] :代表在没有限制的情况下，第i位开始到最低位能选择的数字个数
     */
    int[] dp;
    int[] cnt;

    /**
     * @param now   当前位置
     * @param limit 当前位置有没有限制
     * @return 能选择的数字个数 和 数字1出现的个数
     */
    int[] dfs(int now, boolean limit) {
        // 所有位置都选完了
        if (now == num.length) {
            return new int[]{1, 0};
        }
        if (!limit && dp[now] != -1) {
            return new int[]{cnt[now], dp[now]};
        }
        int sum = 0;
        int count = 0;
        int high = limit ? num[now] - '0' : 9;
        for (int i = 0; i <= high; ++i) {
            int[] t = dfs(now + 1, limit && i == high);
            if (i == 1) {
                sum += t[0];
            }
            sum += t[1];
            count += t[0];
        }
        if (!limit) {
            dp[now] = sum;
            cnt[now] = count;
        }
        return new int[]{count, sum};
    }

    public int countDigitOne(int n) {
        if (n == 0) {
            return 0;
        }
        num = Integer.toString(n).toCharArray();
        dp = new int[num.length];
        cnt = new int[num.length];
        Arrays.fill(dp, -1);
        Arrays.fill(cnt, -1);
        return dfs(0, true)[1];
    }

    void main(Scanner sc) {
        System.out.println(countDigitOne(sc.nextInt()));
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        new Main().main(sc);
    }
}
```

### 最小覆盖子串

题目链接：[U401387 最小覆盖子串 - 洛谷](https://www.luogu.com.cn/problem/U401387)

题目来源：[76. 最小覆盖子串 - 力扣](https://leetcode.cn/problems/minimum-window-substring/description/)

知识点：双指针（滑动窗口）

具体参见[力扣官方题解](https://leetcode.cn/problems/minimum-window-substring/solutions/257359/zui-xiao-fu-gai-zi-chuan-by-leetcode-solution/)

```java
import java.util.Scanner;

public class Main {
    public String minWindow(String s, String t) {
        int[] need = new int[128], cnt = new int[128];
        int needTotal = 0;
        for (char c : t.toCharArray()) {
            if (need[c] == 0) {
                ++needTotal;
            }
            ++need[c];
        }
        int n = s.length(), currentTotal = 0;
        String ans = null;
        // [left, right] 是最小的以right为结尾的可覆盖t的子串
        for (int right = 0, left = 0; right < n; ++right) {
            char ch = s.charAt(right);
            ++cnt[ch];
            if (cnt[ch] == need[ch]) {
                ++currentTotal;
            }
            // 逐步收缩左区间，找到最小的可以覆盖t的子串
            while (left < right) {
                char c = s.charAt(left);
                // 如果收缩左区间会使得对应字符个数不能达到要求，则这就是最小的
                if (cnt[c] - 1 < need[c]) {
                    break;
                }
                --cnt[c];
                ++left;
            }
            if (currentTotal == needTotal && (ans == null || ans.length() > right - left)) {
                ans = s.substring(left, right + 1);
            }
        }
        return ans == null ? "-1" : ans;
    }

    void main(Scanner sc) {
        int t = sc.nextInt();
        for (int i = 0; i < t; i++) {
            System.out.println(minWindow(sc.next(), sc.next()));
        }
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        new Main().main(sc);
    }
}
```

### 用邮票贴满网格图

题目链接：[U401424 用邮票贴满网格图 - 洛谷](https://www.luogu.com.cn/problem/U401424)

题目来源：[2132. 用邮票贴满网格图 - 力扣](https://leetcode.cn/problems/stamping-the-grid/description/)

知识点：前缀和、差分

具体参见[灵茶山艾府的题解](https://leetcode.cn/problems/stamping-the-grid/solutions/1199642/wu-nao-zuo-fa-er-wei-qian-zhui-he-er-wei-zwiu/)

```java
import java.util.Scanner;

public class Main {
    /**
     * idea:<BR>
     * 二维前缀和 & 二维差分<BR>
     * 所有位置能放邮票则放，放了则区间加1<BR>
     * 最后如果存在不能放邮票的地方，则返回false<BR>
     * 在最后判断单个位置是否为能放邮票，用二维差分求出单个格子的值
     */
    public boolean possibleToStamp(int[][] grid, int stampHeight, int stampWidth) {
        int n = grid.length, m = grid[0].length;
        /*
            sum[i][j]   表示 左上角为(0,0) 右下角为 (i-1,j-1) 的矩阵的和
            sum 用来判断能不能放邮票
         */
        int[][] sum = new int[n + 1][m + 1];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                sum[i + 1][j + 1] = sum[i][j + 1] + sum[i + 1][j] - sum[i][j] + grid[i][j];
            }
        }

        int[][] diff = new int[n + 2][m + 2];
        // 尝试在 左上角为(i,j) 右下角为 (i+width-1, j+height-1) 的矩阵内放置邮票
        for (int i = 0; i + stampHeight - 1 < n; ++i) {
            for (int j = 0; j + stampWidth - 1 < m; ++j) {
                // 判断范围内 是否存在 被占据的格子
                if (sum[i + stampHeight][j + stampWidth] + sum[i][j] - sum[i + stampHeight][j] - sum[i][j + stampWidth] == 0) {
                    // 在这个范围内放置邮票(即区间+1)
                    diff[i + 1][j + 1] += 1;
                    diff[i + stampHeight + 1][j + 1] -= 1;
                    diff[i + 1][j + stampWidth + 1] -= 1;
                    diff[i + stampHeight + 1][j + stampWidth + 1] += 1;
                }
            }
        }
        // 计算单个位置的值
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= m; j++) {
                diff[i][j] += diff[i - 1][j] + diff[i][j - 1] - diff[i - 1][j - 1];
                if (diff[i][j] == 0 && grid[i - 1][j - 1] == 0) {
                    return false;
                }
            }
        }
        return true;
    }

    void main(Scanner sc) {
        int test = sc.nextInt();
        for (int t = 0; t < test; t++) {
            int n = sc.nextInt(), m = sc.nextInt(), height = sc.nextInt(), width = sc.nextInt();
            int[][] grid = new int[n][m];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    grid[i][j] = sc.nextInt();
                }
            }
            if (possibleToStamp(grid, height, width)) {
                System.out.println("Yes");
            } else {
                System.out.println("No");
            }
        }
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        new Main().main(sc);
    }
}
```
