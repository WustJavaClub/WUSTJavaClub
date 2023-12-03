## 双指针

[283. 移动零 - 力扣（LeetCode）](https://leetcode.cn/problems/move-zeroes/)

**思路：读写指针**

由于数组是从左到右遍历的，移动0会覆盖后面的元素，这时可以反过来想，被移动的元素是不会被覆盖的，可以移动非零的元素到前面

[75. 颜色分类 - 力扣（LeetCode）](https://leetcode.cn/problems/sort-colors/description/)

思路：

如果设置两个指针，一个0指针，初始化为0，一个2指针，初始化为n-1，每次遇到0就放前面，每次遇到2就放后面，但这样是不行的，因为2会覆盖后面的元素，让后面的元素不可读，因此考虑交换而不是覆盖。如果每次碰到2就与index2交换位置（注意这里要使用循环，因为交换过来的元素可能还是2），那么数组最后就会变成所有的2排在最后，0和1还没有排序，就转化为了类似上一题的思路。

```java
public void swap(int[] nums,int i,int j){
    int t=nums[i];
    nums[i]=nums[j];
    nums[j]=t;
}
public void sortColors(int[] nums) {
    int n=nums.length,index2=n-1;
    for(int i=0;i<=index2;++i){
        while(i<=index2&&nums[i]==2)
            swap(nums,i,index2--);
    }
    int index0=0;
    for(int i=0;i<=index2;++i)
        if(nums[i]==0)
            nums[index0++]=0;
    for(int i=index0;i<=index2;++i)
        nums[i]=1;
}
```

其实下面的两个for循环可以放在第一个for循环里面，因为此时在i以及之前都满足只有0和1，当i这一位是0，就把他交换到index0即可

改进如下：

```java
public void swap(int[] nums,int i,int j){
    int t=nums[i];
    nums[i]=nums[j];
    nums[j]=t;
}
public void sortColors(int[] nums) {
    int index0=0,n=nums.length,index2=n-1;
    for(int i=0;i<=index2;++i){
        while(i<=index2&&nums[i]==2)
            swap(nums,i,index2--);
        if(nums[i]==0)//思考为什么是if而不是while，交换后的结果也可能是0
         swap(nums,i,index0++);
    }
}
```

思考：

index0可能原来就指向0，交换过来了还是0，就可能造成这种局面00110......22222.....，因为经过了while循环，i不可能指向2，所以i总是会遍历到1后面的所有0

应用：

三路快排

```
Random random=new Random();
public void swap(int[] nums,int i,int j){
    int t=nums[i];
    nums[i]=nums[j];
    nums[j]=t;
}
public void quickSort(int[] nums,int left,int right){
    if(left>=right)
        return;
    int index=random.nextInt(right-left+1)+left,standard=nums[index];
    int l=left,r=right;
    for(int i=left;i<=r;++i){
        while(i<=r&&nums[i]>standard)
            swap(nums,i,r--);
        if(nums[i]<standard)
            swap(nums,i,l++);
    }
    quickSort(nums,left,l-1);
    quickSort(nums,r+1,right);
}
public int[] sortArray(int[] nums) {
    int i,n=nums.length;
    quickSort(nums,0,n-1);
    return nums;
}
```

## 前缀和&滑动窗口

**题目出现连续子数组就要想到可能可以用前缀和或者滑动窗口做**

[209. 长度最小的子数组 - 力扣（LeetCode）](https://leetcode.cn/problems/minimum-size-subarray-sum/description/)

本题既可以用滑动窗口也可以用前缀和

前缀和：

大前缀和-小前缀和即为连续子数组总和

TreeMap做法

```java
public int minSubArrayLen(int target, int[] nums) {
    int i,n=nums.length,res=n+1;
    long[] preSum=new long[n+1];
    TreeMap<Long,Integer> map=new TreeMap<>();
    map.put(0L,0);
    for(i=1;i<=n;++i){
        preSum[i]=preSum[i-1]+nums[i-1];
        if(preSum[i]>=target){
            Long key=map.floorKey(preSum[i]-target);
            if(key!=null)
                res=Math.min(res,i-map.get(key));
        }
        map.put(preSum[i],i);
    }
    return res==n+1?0:res;
}
```

二分查找做法

```
public int binarySearch(long[] nums,long target,int left,int right){
    while(left<right){
        int mid=(left+right+1)>>1;
        if(nums[mid]<=target)
            left=mid;
        else
            right=mid-1;
    }
    return nums[left]<=target?left:-1;
}
public int minSubArrayLen(int target, int[] nums) {
    int i,n=nums.length,res=n+1;
    long[] preSum=new long[n+1];
    for(i=1;i<=n;++i){
        preSum[i]=preSum[i-1]+nums[i-1];
        if(preSum[i]>=target){
            int index=binarySearch(preSum,preSum[i]-target,0,i-1);
            if(index!=-1)
                res=Math.min(res,i-index);
        }
    }
    return res==n+1?0:res;
}
```

滑动窗口做法：

技巧，右边界每次扩张1，而左边界可以任意次收缩直到不满足条件

```
public int minSubArrayLen(int target, int[] nums) {
    int left=0,right,n=nums.length,sum=0,res=n+1;
    for(right=0;right<n;++right){
        sum+=nums[right];
        while(sum>=target){
            res=Math.min(res,right-left+1);
            sum-=nums[left++];
        }
    }
    return res==n+1?0:res;
}
```

[238. 除自身以外数组的乘积 - 力扣（LeetCode）](https://leetcode.cn/problems/product-of-array-except-self/)

思路：

前后缀乘积，优化是看0的个数，超过2直接返回0数组，只有1个则只算0位置的

```
public int[] productExceptSelf(int[] nums) {
    int i,n=nums.length,count=0,index=-1;
    int[] res=new int[n];
    for(i=0;i<n;++i)
        if(nums[i]==0){
            ++count;
            index=i;
            if(count>1)
                return res;
        }
    if(count==1){
        int t=1;
        for(i=0;i<index;++i)
            t*=nums[i];
        for(i=index+1;i<n;++i)
            t*=nums[i];
        res[index]=t;
        return res;
    }else{
        int[] pre=new int[n+1],sub=new int[n+1];
        pre[0]=1;
        sub[n]=1;
        for(i=1;i<=n;++i)
            pre[i]=pre[i-1]*nums[i-1];
        for(i=n-1;i>=0;--i)
            sub[i]=sub[i+1]*nums[i];
        for(i=0;i<n;++i)
            res[i]=sub[i+1]*pre[i];
        return res;
    }
}
```

## 差分

[1109. 航班预订统计 - 力扣（LeetCode）](https://leetcode.cn/problems/corporate-flight-bookings/)

与前缀和相对的概念

## 单调栈

[739. 每日温度 - 力扣（LeetCode）](https://leetcode.cn/problems/daily-temperatures/)

## 单调队列

[239. 滑动窗口最大值 - 力扣（LeetCode）](https://leetcode.cn/problems/sliding-window-maximum/description/)

二分答案

要求：满足单调性（极小化极大），知道选择的策略（有显然的顺序，而不是需要思考顺序，或者不需要考虑顺序）

[LCR 073. 爱吃香蕉的狒狒 - 力扣（LeetCode）](https://leetcode.cn/problems/nZZqjQ/)

[793. 阶乘函数后 K 个零 - 力扣（LeetCode）](https://leetcode.cn/problems/preimage-size-of-factorial-zeroes-function/)
