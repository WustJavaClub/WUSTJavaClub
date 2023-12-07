---
author : Cattle_Horse
create : 2023/11/11
---

# 第四次作业报告

姓名：张伟志

班级：软件2102

## 2917. 找出数组中的 K-or 值

题目链接：[2917. 找出数组中的 K-or 值 - 力扣](https://leetcode.cn/problems/find-the-k-or-of-an-array/description/)

时间复杂度：$\Theta(n\log m)$ 其中 $m$ 为数组中元素的最大值

空间复杂度：$\Theta(1)$

数据不超过二进制 $31$ 位，枚举 $31$ 位二进制，取出数组中每一个数的对应二进制位，如果这些二进制位中 $1$ 的个数大于 $k$，则将结果的这意味置为 $1$

```java
public class Problem2917 {
    public int findKOr(int[] nums, int k) {
        int ans = 0;
        for (int i = 0; i < 32; i++) {
            int cnt = 0;
            for (int num : nums) {
                cnt += num >> i & 1;
            }
            if (cnt >= k) {
                ans |= 1 << i;
            }
        }
        return ans;
    }
}
```

## 268. 丢失的数字

题目链接：[268. 丢失的数字 - 力扣](https://leetcode.cn/problems/missing-number/description/)

### 方法一

时间复杂度：$\Theta(n)$

空间复杂度：$\Theta(n)$

将数组中 $ n$ 个数均加入 HashSet 中，再遍历 $[0,n]$ 判断 HashSet 中是否该数，若不存在则就是答案

```java
public int missingNumber(int[] nums) {
    Set<Integer> set = new HashSet<Integer>();
    int n = nums.length;
    for (int i = 0; i < n; i++) {
        set.add(nums[i]);
    }
    for (int i = 0; i <= n; i++) {
        if (!set.contains(i)) {
            return i;
        }
    }
    return -1;
}
```

### 方法二

时间复杂度：$\Theta(n)$

空间复杂度：$\Theta(1)$

计算 $[0,n]$ 的和，减去数组中的每个数，剩下的就是答案

```java
public int missingNumber(int[] nums) {
    int ans = 0, n = nums.length;
    for (int i = 0; i < n; i++) {
        ans += i - nums[i];
    }
    return ans + n;
}
```

### 方法三

时间复杂度：$\Theta(n)$

空间复杂度：$\Theta(1)$

在 $[0,n]$ 与 数组中的元素中，除了没有在数组中出现的元素，其他均出现了两次，利用异或两两抵消的性质求出答案

```java
public int missingNumber(int[] nums) {
    int ans = 0, n = nums.length;
    for (int i = 0; i < n; i++) {
        ans = ans ^ i ^ nums[i];
    }
    return ans ^ n;
}
```

## 35. 搜索插入位置

题目链接：[35. 搜索插入位置 - 力扣](https://leetcode.cn/problems/search-insert-position/)

时间复杂度：$\Theta(\log n)$

空间复杂度：$\Theta(1)$

二分查找

- 如果当前指向元素大于目标元素，则说明应该在当前指向的左侧或当前位置插入（因为可能没有更小的了）
- 如果当前指向元素等于目标元素，返回当前索引
- 如果当前指向元素小于目标元素，则说明应该在当前指向的右侧插入

```java
public class Problem35 {
    public int searchInsert(int[] nums, int target) {
        // 可能插入到末尾，但由于下取整，不需要进行越界判断
        int left = 0, right = nums.length;
        while (left < right) {
            int mid = left + right >> 1;
            if (nums[mid] > target) {
                right = mid;
            } else if (nums[mid] == target) {
                return mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
}
```

## 69. x 的平方根

题目链接：[69. x 的平方根 - 力扣](https://leetcode.cn/problems/sqrtx/description/)

时间复杂度：$\Theta(\log n)$

空间复杂度：$\Theta(1)$

二分查找

- 如果 $mid$ 的平方大于 $x$，则说明结果在左侧
- 如果 $mid$ 的平方等于 $x$，则返回 $mid$
- 如果 $mid$ 的平方小于 $x$，则说明结果在右侧或 $mid$

注意：

1. 取 $mid$ 时可能整型溢出；
2. 区间收缩时，其中一个分支的左端点仍为 $mid$，下取整会死循环，采用上取整

```java
public class Problem69 {
    public int mySqrt(int x) {
        int left = 0, right = x;
        while (left < right) {
            int mid = left + right + 1 >>> 1;
            long square = (long) mid * mid;
            if (square > x) {
                right = mid - 1;
            } else if (square == x) {
                return mid;
            } else {
                left = mid;
            }
        }
        return left;
    }
}
```

## 338. 比特位计数

题目链接：[338. 比特位计数 - 力扣](https://leetcode.cn/problems/counting-bits/description/)

### 方法一

时间复杂度：$\Theta(n\log n)$

空间复杂度：$\Theta(1)$（不考虑答案数组）

二进制计算每一个数的二进制中 $1$ 的个数

- `x & (x - 1)`：将二进制中的最后一个 $1$ 置为 $0$
- `x & -x`：取出二进制中的最后一个 $1$

```java
class Solution1 {
    public int[] countBits(int n) {
        int[] ans = new int[n + 1];
        for (int i = 0; i <= n; i++) {
            ans[i] = bitCount(i);
        }
        return ans;
    }

    private int bitCount(int x) {
        int ans = 0;
        while (x != 0) {
            x &= x - 1;
            // x -= x & -x;
            ++ans;
        }
        return ans;
    }
}
```

### 方法二

时间复杂度：$\Theta(n)$

空间复杂度：$\Theta(1)$（不考虑答案数组）

对于一个数 $x$：

- 如果它的二进制位的最低位是 $0$，则其二进制中 $1$ 的个数与 `x >> 1` 相同
- 如果它的二进制位的最低位是 $0$，则其二进制中 $1$ 的个数比 `x >> 1` 多 $1$

$$
f(x) = \begin{cases}
	f(\lfloor\dfrac{x}{2}\rfloor) & \text{ if } x \& 1 = 0\\
	f(\lfloor\dfrac{x}{2}\rfloor) + 1 & \text{ if } x \& 1 = 1
\end{cases}
$$

```java
class Solution2 {
    public int[] countBits(int n) {
        int[] dp = new int[n + 1];
        for (int i = 0; i <= n; ++i) {
            if ((i & 1) == 0) {
                dp[i] = dp[i >> 1];
            } else {
                dp[i] = dp[i >> 1] + 1;
            }
        }
        return dp;
    }
}
```

## 477. 汉明距离总和

题目链接：[477. 汉明距离总和](https://leetcode.cn/problems/total-hamming-distance/)

### 方法一

时间复杂度：$\Theta(n^2\log m)$ 其中 $\log m$ 是数组中数据的二进制位数

空间复杂度：$\Theta(1)$

循环枚举两个数，再求两个数的汉明距离（通过异或找出不同再判断二进制 $1$ 的个数）

```java
class Solution1 {
    public int totalHammingDistance(int[] nums) {
        int n = nums.length, ans = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                ans += Integer.bitCount(nums[i] ^ nums[j]);
            }
        }
        return ans;
    }
}
```

### 方法二

时间复杂度：$\Theta(n\log m)$ 其中 $\log m$ 是数组中数据的二进制位数

空间复杂度：$\Theta(1)$

计算每一个二进制位的不同的数量，以最低位举例（$n$ 为数组长度）：

如果数组中二进制最低位为 $1$ 的个数是 $x$，则 $0$ 的个数就是 $n - x$，每一个 $1$ 可以和 $n-x$ 个 $0$ 进行组合，使得答案增加 $n-x$。

即每一位置可以使得答案增加 $x\times(n-x)$

```java
class Solution2 {
    public int totalHammingDistance(int[] nums) {
        int ans = 0, n = nums.length;
        for (int i = 0; i < 32; i++) {
            int x = 0;
            for (int num : nums) {
                x += num >> i & 1;
            }
            ans += x * (n - x);
        }
        return ans;
    }
}
```

## 784. 字母大小写全排列

题目链接：[784. 字母大小写全排列](https://leetcode.cn/problems/letter-case-permutation/description/)

### 方法一

时间复杂度：$\Theta(n\times2^n)$ 其中 $n$ 是给定字符串的长度

空间复杂度：$\Theta(2^n)$ 递归栈需要的空间

递归每一个位置

```java
class Solution1 {
    private int n;
    private char[] str;
    private ArrayList<String> ans;

    private void dfs(int index) {
        if (index == n) {
            ans.add(String.valueOf(str));
            return;
        }
        if (Character.isDigit(str[index])) {
            dfs(index + 1);
        } else {
            dfs(index + 1);
            str[index] ^= 32;
            dfs(index + 1);
        }
    }

    public List<String> letterCasePermutation(String s) {
        n = s.length();
        str = s.toCharArray();
        ans = new ArrayList<>();
        dfs(0);
        return ans;
    }
}
```

### 方法二

时间复杂度：$\Theta(n\times2^n)$ 其中 $n$ 是给定字符串的长度

空间复杂度：$\Theta(1)$

二进制枚举子集（屏蔽是数字的二进制位）

```java
class Solution2 {
    public List<String> letterCasePermutation(String str) {
        int mask = 0;
        int n = str.length();
        for (int i = 0; i < n; i++) {
            if (Character.isLetter(str.charAt(i))) {
                mask |= 1 << i;
            }
        }
        // 二进制中为1的位置表示进行大小写转换
        // 这个集合的子集
        int s = mask;
        ArrayList<String> ans = new ArrayList<>(1 << (Integer.bitCount(mask)));
        do {
            StringBuilder current = new StringBuilder(n);
            for (int i = 0; i < n; i++) {
                if ((s >> i & 1) == 1) {
                    current.append((char) (str.charAt(i) ^ 32));
                } else {
                    current.append(str.charAt(i));
                }
            }
            ans.add(current.toString());
            s = (s - 1) & mask;
        } while (s != mask);
        return ans;
    }
}
```

## 90. 子集 II

题目链接：[90. 子集 II](https://leetcode.cn/problems/subsets-ii/description/)

### 方法一

时间复杂度：$\Theta(n\times 2^n)$

空间复杂度：$\Theta(2^n)$

递归枚举子集

如何去重：排序后，跳过相同元素的枚举

```java
class Solution1 {
    int[] nums;
    int n;
    List<List<Integer>> ans;

    void dfs(int index, ArrayList<Integer> current) {
        if (index == n) {
            ans.add(List.copyOf(current));
            return;
        }
        // 选择当前数字
        current.add(nums[index]);
        dfs(index + 1, current);

        // 不选择当前数字
        current.remove(current.size() - 1);
        // 跳过相同数字，再选择当前数字时会计算这些情况
        while (index + 1 < n && nums[index] == nums[index + 1]) {
            ++index;
        }
        dfs(index + 1, current);
    }

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        ans = new ArrayList<>();
        this.nums = nums;
        n = nums.length;
        Arrays.sort(nums);
        dfs(0, new ArrayList<>(n));
        return ans;
    }
}
```

### 方法二

时间复杂度：$\Theta(n\times 2^n)$

空间复杂度：$\Theta(n)$

[CopyRight LeetCode](https://leetcode.cn/problems/subsets-ii/solutions/690549/zi-ji-ii-by-leetcode-solution-7inq/)

> 考虑数组 $[1, 2, 2]$，选择前两个数，或者第一、三个数，都会得到相同的子集。
>
> 也就是说，对于当前选择的数 $x$，若前面右与其相同的数 $y$，且没有选择 $y$，此时包含 $x$ 的子集，必然会出现在包含 $y$ 的所有子集中

```java
class Solution2 {
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        int len = nums.length;
        List<List<Integer>> ans = new ArrayList<>(1 << len);
        for (int i = 0, S = 1 << len; i < S; ++i) {
            List<Integer> t = new ArrayList<>();
            boolean mark = true;
            for (int j = 0; j < len; ++j) {
                if ((i >> j & 1) == 1) {
                    if (j > 0 && nums[j] == nums[j - 1] && (i >> (j - 1) & 1) == 0) {
                        mark = false;
                        break;
                    }
                    t.add(nums[j]);
                }
            }
            if (mark) {
                ans.add(t);
            }
        }
        return ans;
    }
}
```

## 793. 阶乘函数后 K 个零

题目链接：[793. 阶乘函数后 K 个零](https://leetcode.cn/problems/preimage-size-of-factorial-zeroes-function/description/)

时间复杂度：$\Theta(\log n)$

空间复杂度：$\Theta(1)$

阶乘函数后的 $0$ 由 $2\times 5$ 生成，因此其个数取决于阶乘中因子 $2$ 和 $5$ 的个数（取较小的那个），而阶乘中 $5$ 的个数一定小于 $2$

二分查找上界和下界

```java
class Solution {
    /**
     * Get the number of factors that are f in the factorial of element
     *
     * @param element a long
     * @param f       a factor
     * @return the number of factors that are f in the factorial of element
     */
    int getF(long element, int f) {
        int ans = 0;
        while (element > 0) {
            ans += element / f;
            element /= f;
        }
        return ans;
    }

    /**
     * Get the first number that satisfies the factorial followed by 0's with k
     * count(getF(element,5)) >= k, the first element
     */
    int getLower(int k) {
        long left = 0, right = (long) 5e9;
        while (left < right) {
            long mid = left + right >> 1;
            int count = getF(mid, 5);
            if (count >= k) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return getF(left, 5) == k ? (int) left : -1;
    }

    /**
     * Get the last number that satisfies the factorial followed by 0's with k
     * count(getF(element,5)) >= k, the last element
     */
    int getUpper(int k) {
        long left = 0, right = (long) 5e9;
        while (left < right) {
            long mid = left + right + 1 >> 1;
            int count = getF(mid, 5);
            if (count > k) {
                right = mid - 1;
            } else {
                left = mid;
            }
        }
        return getF(left, 5) == k ? (int) left : -1;
    }

    public int preimageSizeFZF(int k) {
        int upper = getUpper(k);
        if (upper == -1) {
            return 0;
        }
        int lower = getLower(k);
        return upper - lower + 1;
    }
}
```

## 982. 按位与为零的三元组

题目链接：[982. 按位与为零的三元组](https://leetcode.cn/problems/triples-with-bitwise-and-equal-to-zero/description/)

时间复杂度：$\Theta(n^2+2^{16}n)$

空间复杂度：$\Theta(2^{16})$

和零比较的话会有三个变量一个定量，式子转换会变为只有三个变量

`nums[i] & nums[j] & nums[k] == 0`  $\Rightarrow$ `F( nums[i] & nums[j], nums[k] ) ` ，其中 $F(a,b)$ 表示 $a$ 与 $b$ 没有同时为 $1$ 的二进制位

先确定 $a$ 和 $b$ 的其中一个，再对其补集进行枚举子集

```java
public class Problem982 {
    /**
     * 全集
     */
    private static final int U = 0xffff;

    public int countTriplets(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<>(1 << 16);
        for (int i : nums) {
            for (int j : nums) {
                map.merge(i & j, 1, Integer::sum);
            }
        }
        int ans = 0;
        for (int num : nums) {
            // 求补集
            int complement = num ^ U;
            // 枚举这个补集的子集
            int s = complement;
            do {
                ans += map.getOrDefault(s, 0);
                s = (s - 1) & complement;
            } while (s != complement);
        }
        return ans;
    }
}
```

## 1178. 猜字谜

题目链接：[1178. 猜字谜](https://leetcode.cn/problems/number-of-valid-words-for-each-puzzle/description/)

时间复杂度：xxx

空间复杂度：xxx

[CopyRight LeetCode](https://leetcode.cn/problems/number-of-valid-words-for-each-puzzle/solutions/622145/cai-zi-mi-by-leetcode-solution-345u/)

```java
/**
 * @author : Cattle_Horse
 * @date : 2023/02/19 20:38
 * @description : <a href="https://leetcode.cn/problems/number-of-valid-words-for-each-puzzle/submissions/403939784/">1178.猜字谜</a>
 **/
class Solution {
    public List<Integer> findNumOfValidWords(String[] words, String[] puzzles) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (String s : words) {
            // 二进制映射
            int mask = 0;
            for (int i = 0; i < s.length(); ++i) {
                mask |= 1 << (s.charAt(i) - 'a');
            }
            // 题目保证puzzle字符串长度为7
            // 只加入个数小于等于7的减少空间消耗
            if (Integer.bitCount(mask) <= 7) {
                map.put(mask, map.getOrDefault(mask, 0) + 1);
            }
        }
        List<Integer> ans = new ArrayList<>(puzzles.length);
        for (String s : puzzles) {
            // 二进制映射
            int mask = 0;
            // 跳过首字母，之后处理集合的时候单独加上，保证首字母存在
            for (int i = 1; i < s.length(); ++i) {
                mask |= 1 << (s.charAt(i) - 'a');
            }
            int cnt = 0;
            int begin = s.charAt(0) - 'a';
            for (int i = mask; i != 0; i = (i - 1) & mask) {
                // 保证首字母存在
                cnt += map.getOrDefault(i | (1 << begin), 0);
            }
            // 处理空集（只有首字母的情况）
            cnt += map.getOrDefault(1 << begin, 0);
            ans.add(cnt);
        }
        return ans;
    }
}
```

## U378574 Palindrome Pairs

题目链接：[U378574 Palindrome Pairs](https://www.luogu.com.cn/problem/U378574)

两个字符串若能任意排列成回文串，则出现的相同的字母个数至多只能有一个是奇数，即：

- $a\sim z$ 中，均出现了偶数次
- $a\sim z$ 中，只有一个字母出现了奇数次

将字符串转化为 $26$ 位的二进制串，最低位表示 $a$ 出现了的是奇数次还是偶数次

对于两个 $01$ 串 $A$ 和 $B$

- 若要构成回文串，有以下情况：

- $A \oplus B=0$ 即 $A=B$

- $A \oplus B$ 的二进制 $1$ 的个数为 $0$



输入部分

```java
public class U378574 {
    void main(Scanner sc) {
        new Solution1().main(sc);
        new Solution2().main(sc);
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        new U378574().main(sc);
    }

    /**
     * 将字符串中字符个数转化为 01 串<p>
     * 如 aabbbc 会返回 (110)_2
     */
    public static int getMask(String str) {
        int mask = 0;
        for (char ch : str.toCharArray()) {
            mask ^= 1 << (ch - 'a');
        }
        return mask;
    }
}
```

### 方法一（超时）

时间复杂度：$\Theta(n^2\log n)$

空间复杂度：$\Theta(1)$ 不考虑输入数组占用空间

两重循环枚举两个字符串，将字符串转化为二进制，判断这个二进制数是否只有 $1$ 个或没有 $1$

```java
class Solution1 {
    /**
     * 判断两个字符串能否构成回文串
     */
    boolean check(String str1, String str2) {
        int mask = U378574.getMask(str1) ^ U378574.getMask(str2);
        return mask == 0 || ((mask & (mask - 1)) == 0);
    }

    public void main(Scanner sc) {
        int n = sc.nextInt();
        String[] str = new String[n];
        for (int i = 0; i < n; i++) {
            str[i] = sc.next();
        }
        long ans = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (check(str[i], str[j])) {
                    ++ans;
                }
            }
        }
        System.out.println(ans);
    }
}
```

### 方法二

时间复杂度：$\Theta(n\log n)$

空间复杂度：$\Theta(n)$ 不考虑输入数组占用空间

HashMap 记录每一种 $01$ 串出现的次数，枚举每一个 $01$ 串，找这个 $01$ 串能够和多少 $01$ 串组合成回文串，详见代码

```java
class Solution2 {
    public void main(Scanner sc) {
        HashMap<Integer, Integer> map = new HashMap<>();
        int n = sc.nextInt();
        for (int i = 0; i < n; i++) {
            int mask = U378574.getMask(sc.next());
            map.merge(mask, 1, Integer::sum);
        }
        long ans = 0;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            int mask = entry.getKey(), count = entry.getValue();
            // 01 串相同的部分
            ans += (long) count * (count - 1) / 2;

            // 只有一个 1 的情况
            for (int i = 0; i < 26; ++i) {
                // 为防止重复，只计算 1变0 或者 0变1 中的一种情况
                if ((mask >> i & 1) == 0) {
                    continue;
                }
                int flipOneBinaryBit = mask ^ (1 << i);
                int count2 = map.getOrDefault(flipOneBinaryBit, 0);
                ans += (long) count * count2;
            }
        }
        System.out.println(ans);
    }
}
```
