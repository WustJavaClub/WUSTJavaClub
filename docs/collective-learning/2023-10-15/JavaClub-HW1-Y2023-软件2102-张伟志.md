# 第一次作业报告

姓名：张伟志

班级：软件2102

## TODO

- [ ] 判断【001. 3 或 5 的倍数】结果范围
- [ ] 补充【004. 最大回文数乘积】方法二 思路 及 时间复杂度

## 001. 3 或 5 的倍数

题目链接：[U264430 001. 3 或 5 的倍数](https://www.luogu.com.cn/problem/U264430)

### 方法一

时间复杂度：$O( n )$

空间复杂度：$O(1)$

#### 思路

循环遍历题目所给范围中的每一个数字 是否满足“能被 $3$ 或 $5$ 整除”这个条件

数据范围最大为 $10^8$，由于是求满足条件的数字的和，结果可能会出现 $int$ 整型的溢出，所以选择 $long$ 类型尝试

#### 代码

```java
import java.util.Scanner;

public class Main {
    /**
     * @param x 待判断的数字
     * @return 参数x是否能被3或5整除
     */
    boolean check(int x) {
        return x % 3 == 0 || x % 5 == 0;
    }

    void main(Scanner sc) {
        int n = sc.nextInt();
        long ans = 0;
        for (int i = 2; i < n; ++i) {
            if (check(i)) {
                ans += i;
            }
        }
        System.out.println(ans);
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        new Main().main(sc);
    }
}

```

### 方法二

时间复杂度：$O(1)$

空间复杂度：$O(1)$

#### 思路

用 $n$ 来表示右边界（可取，与题目不同）

用 $3x$ 来表示能被 $3$ 整除的数字，其中 $x\le \left \lfloor \dfrac{n}{3} \right \rfloor $

所以在 $[1,n]$ 中，有 $\left \lfloor \dfrac{n}{3} \right \rfloor$ 个能被 $3$ 整除的数字，它们分别是 $3\times 1,3\times 2,\dots,3\times\left \lfloor \dfrac{n}{3} \right \rfloor$

根据等差数列求和公式 $S_n=\dfrac{(a_1+a_n)\times n}{2}$ 可以求出 $[1,n]$ 中能被 $3$ 整除的数字的和

同理可以求出 $[1,n]$ 中能被 $k$ 整除的数字的和为 $\dfrac{(1+\left \lfloor \frac{n}{k} \right \rfloor)\times \left \lfloor \frac{n}{k} \right \rfloor \times k}{2}$

此时可以 $O(1)$ 地求出  $[1,n]$ 中能被 $k$ 整除的数字的和

但是在 $[1,n]$ 中，同时为 $3$ 和 $5$ 的倍数的数字会被多减去一次（[容斥原理](https://oi-wiki.org//math/combinatorics/inclusion-exclusion-principle/)），如 $15,30$

因此再加上 $[1,n]$ 中能被 $15$ 整除的数字的和

PS：这个方法需要注意计算和的过程中是否会数值溢出，可以考虑使用 $BigInteger$ 高精度整型类

#### 代码

```java
import java.util.Scanner;

public class Main {
    /**
     * @param n         右边界
     * @param factor    被整除的数
     * @return          [1, n] 中 能被 factor 整除的数的和
     */
    long getSum(int n, int factor) {
        int num = n / factor;
        return (1L + num) * num / 2 * factor;
    }

    void main(Scanner sc) {
        int n = sc.nextInt() - 1;
        long ans = getSum(n, 3) + getSum(n, 5) - getSum(n, 15);
        System.out.println(ans);
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        new Main().main(sc);
    }
}

```

## 002. 偶数斐波那契数

题目链接：[U264934 002. 偶数斐波那契数](https://www.luogu.com.cn/problem/U264934)

### 方法一

时间复杂度：$O( n )$

空间复杂度：$O(1)$

#### 思路

使用斐波那契数列递推式 $fib_n=fib_{n-1}+fib_{n-2}$

用数组存储结果，递推迭代时发现计算第 $k$ 项只用到了 $k-1$ 和 $k-2$ 项的值，采用滚动数组重复利用空间

#### 代码

```java
import java.util.Scanner;

public class Main {
    void main(Scanner sc) {
        int n = sc.nextInt();
        final int mod = 2;
        int[] fib = new int[mod];
        fib[0 % mod] = 1;
        fib[1 % mod] = 1;
        long ans = 0;
        for (int i = 2; true; ++i) {
            fib[i % mod] += fib[(i - 1) % mod];
            if (fib[i % mod] > n) {
                break;
            }
            if (fib[i % mod] % 2 == 0) {
                ans += fib[i % mod];
            }
        }
        System.out.println(ans);
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        new Main().main(sc);
    }
}
```

#### 感受

题目有点难理解，起初以为这题是求解偶数项位置的斐波那契数的和

## 003. 最大质因数

题目链接：[U264937 003. 最大质因数](https://www.luogu.com.cn/problem/U264937)

### 方法一

时间复杂度：$O(\sqrt{n})$

空间复杂度：$O(1)$

#### 思路

[分解质因数 - OI Wiki](https://oi-wiki.org/math/number-theory/pollard-rho/)

#### 代码

```java
import java.util.Scanner;

public class Main {
    void main(Scanner sc) {
        long n = sc.nextLong();
        long ans = 0;
        for (long factor = 2; factor * factor <= n; ++factor) {
            if (n % factor == 0) {
                ans = factor;
                while (n % factor == 0) {
                    n /= factor;
                }
            }
        }
        System.out.println(Math.max(ans, n));
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        new Main().main(sc);
    }
}
```

## 004. 最大回文数乘积

题目链接：[U264963 004. 最大回文数乘积](https://www.luogu.com.cn/problem/U264963)

### 方法一

时间复杂度：$O(10^{2n}\log n)$

空间复杂度：$O(1)$

#### 思路

两重循环枚举两个数求其乘积，对这个乘积判断是否是回文数

#### 代码

```java
import java.util.Scanner;

public class Main {
    int getReverse(int x) {
        int result = 0;
        while (x != 0) {
            result = result * 10 + x % 10;
            x /= 10;
        }
        return result;
    }

    boolean isPalindrome(int x) {
        return x == getReverse(x);
    }

    void main(Scanner sc) {
        int n = sc.nextInt();
        int ans = 0;
        int end = (int) Math.pow(10, n);
        int start = end / 10;
        for (int i = start; i < end; ++i) {
            for (int j = start; j < end; ++j) {
                int mul = i * j;
                if (isPalindrome(mul)) {
                    ans = Math.max(ans, mul);
                }
            }
        }
        System.out.println(ans);
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        new Main().main(sc);
    }
}
```

### 方法二

时间复杂度：TODO

空间复杂度：$O(1)$

#### 思路

在题目所给出范围的两个数的乘积的范围中，从大到小枚举，进行如下操作：

1. 判断这个数是否是回文数
2. 如果是回文数，判断是否能由题目所给出的位数的数字进行乘积组合
   - 如果能由题目所给出的位数的数字进行乘积组合，则这个数字就是结果
   - 否则继续枚举
3. 如果不是回文数，继续枚举

#### 代码

TODO

## 005. 最小公倍数

题目链接：[U265015 005. 最小公倍数](https://www.luogu.com.cn/problem/U265015)

### 方法一

时间复杂度：$O(n\log n)$

空间复杂度：$O(1)$

#### 思路

[求解多个数的最小公倍数 - OI Wiki](https://oi-wiki.org//math/number-theory/gcd/#多个数)

#### 代码

```java
import java.util.Scanner;

public class Main {
    long gcd(long a, long b) {
        if (b == 0) {
            return a;
        }
        return gcd(b, a % b);
    }

    long lcm(long a, long b) {
        return a / gcd(a, b) * b;
    }

    void main(Scanner sc) {
        int n = sc.nextInt();
        long ans = 1;
        for (int i = 2; i <= n; ++i) {
            ans = lcm(ans, i);
        }
        System.out.println(ans);
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        new Main().main(sc);
    }
}
```

## 011. 网格中的最大乘积

题目链接：[U265346 011. 网格中的最大乘积](https://www.luogu.com.cn/problem/U265346)

### 方法一

时间复杂度：$O(n\times m\times x)$

空间复杂度：$O(n\times m)$

#### 思路

遍历二维网格的每一个格子，对这个格子的 垂直、水平、左下、右下 四个方向的连续 $x$ 个数字的乘积，并求最大值

#### 代码

```java
import java.util.Scanner;

public class Main {
    int[][] arr;
    int n, m, count;

    long getMul(int startX, int startY, int offsetX, int offsetY) {
        int endX = startX + offsetX * (count - 1);
        int endY = startY + offsetY * (count - 1);
        if (endX < 0 || endX >= n || endY < 0 || endY >= m) {
            return -1;
        }
        long mul = arr[startX][startY];
        for (int i = 1; i < count; ++i) {
            startX += offsetX;
            startY += offsetY;
            mul *= arr[startX][startY];
        }
        return mul;
    }

    void main(Scanner sc) {
        n = sc.nextInt();
        m = sc.nextInt();
        count = sc.nextInt();
        arr = new int[n][m];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                arr[i][j] = sc.nextInt();
            }
        }
        long ans = 0;
        // 垂直，水平，左下，右下
        final int[] offsetX = new int[]{1, 0,  1, 1};
        final int[] offsetY = new int[]{0, 1, -1, 1};
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                for (int k = 0; k < 4; ++k) {
                    ans = Math.max(ans, getMul(i, j, offsetX[k], offsetY[k]));
                }
            }
        }
        System.out.println(ans);
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        new Main().main(sc);
    }
}
```

#### 补充

以水平方向为例

当计算第 $1$ 个格子的水平方向连续 $x$ 个数字的乘积时，可以存储下这个乘积

当计算第 $2$ 个格子的水平方向连续 $x$ 个数字的乘积时，可以利用第 $1$ 个格子对应乘积的值，用这个值除去第一个格子本身的值，就可以不用重复计算剩余 $x-1$ 个数字的乘积了

## 012. 高度可除的三角数

题目链接：[U265394 012. 高度可除的三角数](https://www.luogu.com.cn/problem/U265394)

### 方法一

时间复杂度：与第一个三角数的因子大于 $n$ 的三角数值有关

空间复杂度：$O(1)$

#### 思路

[算术基本定理 - 百度百科](https://baike.baidu.com/item/算术基本定理)

> 算术基本定理（唯一分解定理）：
>
> 对于大于 $1$ 的自然数 $n$，可以唯一分解成有限个质数的乘积 $n=p_1^{a_1}\times p_2^{a_2}\times \dots\times p_m^{a_m}$，其中 $p_1<p_2<\dots<p_m$ 均为质数，指数 $a_i$ 是正整数。

$p_i^{a_i}$ 的约数有 $p_i^0,p_i^1,\dots,p_i^{a_i}$ 共 $a_i+1$ 个

所以对于每一个 $p_i^{a_i}$ 均有 $a_i+1$ 个选择

而 $p_i^{a_i}$ 间的任意组合都能构成一个 $n$ 的因子

因此，$n$ 的约数个数等于 $\prod_{i=1}^{m}(a_i+1)$

利用[分解质因数](https://oi-wiki.org/math/number-theory/pollard-rho/)求解

#### 代码

```java
import java.util.Scanner;

public class Main {
    /**
     * @param n 待求因子个数的值
     * @return n的因子个数
     */
    int getSigma0(int n) {
        int ans = 1;
        for (int i = 2; i * i <= n; ++i) {
            if (n % i == 0) {
                // k 质因子i出现的次数
                int k = 1;
                while (n % i == 0) {
                    ++k;
                    n /= i;
                }
                ans *= k;
            }
        }
        // 还剩一个因子
        if (n > 1) {
            ans *= 2;
        }
        return ans;
    }

    void main(Scanner sc) {
        int n = sc.nextInt();
        for (int i = 1, sum = 0; true; ++i) {
            sum += i;
            if (getSigma0(sum) > n) {
                System.out.println(sum);
                return;
            }
        }
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        new Main().main(sc);
    }
}
```

## 014. 最长的考拉兹序列

题目链接：[U265423 014. 最长的考拉兹序列](https://www.luogu.com.cn/problem/U265423)

### 方法一

时间复杂度：$\sum_{i=1}^{n}collatz(i)$，其中 $collatz(i)$ 为计算 $i$ 的 $collatz$ 序列的操作次数

空间复杂度：$O(1)$ 

#### 思路

遍历 $[1,n]$ 的每一个数字，对这些数字计算 $collatz$ 序列长度，并求最大长度与对应的数字

#### 代码

```java
import java.util.Scanner;

public class Main {
    int getCollatz(int n) {
        int count = 1;
        long current = n;
        while (current != 1) {
            if (current % 2 == 0) {
                current /= 2;
            } else {
                current = 3 * current + 1;
            }
            ++count;
        }
        return count;
    }

    void main(Scanner sc) {
        int n = sc.nextInt();
        // ansValue : 所给范围中 Collatz 最长的序列的起始数字
        // ansLength: 所给范围中 Collatz 最长的序列的序列长度
        int ansValue = 0, ansLength = 0;
        for (int i = 1; i <= n; ++i) {
            int length = getCollatz(i);
            if (length > ansLength) {
                ansValue = i;
                ansLength = length;
            }
        }
        System.out.println(ansValue + " " + ansLength);
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        new Main().main(sc);
    }
}
```

#### 补充

其中求 $collatz$ 序列长度可添加记忆化数组，但其大小不确定，只能使用 $HashMap$ 存储，使用后发现情况过多，内存不足
