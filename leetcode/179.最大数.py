#
# @lc app=leetcode.cn id=179 lang=python3
#
# [179] 最大数
#
import random
# @lc code=start

from typing import List



class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        # 定义自定义比较函数：如果 a+b > b+a，则 a 应该排在 b 前面
        def is_greater(a, b):
            return str(a) + str(b) > str(b) + str(a)

        def quick_sort(arr, l, r):
            if l >= r:
                return
            
            # 随机化 pivot 防止最坏情况 O(n^2)
            pivot_idx = random.randint(l, r)
            arr[l], arr[pivot_idx] = arr[pivot_idx], arr[l]
            
            pivot = arr[l]
            i, j = l, r
            
            while i < j:
                # 从右向左：寻找比 pivot "大" 的元素（在我们的规则下）
                # 注意这里用 <= 是为了让相等的元素也能被跳过，防止死循环
                while i < j and not is_greater(arr[j], pivot):
                    j -= 1
                arr[i] = arr[j]
                
                # 从左向右：寻找比 pivot "小" 的元素
                while i < j and is_greater(arr[i], pivot):
                    i += 1
                arr[j] = arr[i]
            
            arr[i] = pivot
            quick_sort(arr, l, i - 1)
            quick_sort(arr, i + 1, r)

        quick_sort(nums, 0, len(nums) - 1)

        # 拼接结果
        res = "".join(map(str, nums))
        
        # 处理特殊情况：例如 [0, 0] 应该返回 "0" 而不是 "00"
        return '0' if res[0] == '0' else res
# @lc code=end

        
# @lc code=end

