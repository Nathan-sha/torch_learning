#
# @lc app=leetcode.cn id=164 lang=python3
#
# [164] 最大间距
#
from typing import List
# @lc code=start
class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return 0
        nums = self.quick_sort(nums)
        res = 0
        for i in range(1, len(nums)):
            temp = nums[i] - nums[i-1]
            res = max(temp, res)
        
        return res


    def sort(self, nums):
        # 冒泡
        for i in range(len(nums)):
            change = False
            for j in range(len(nums)-i-1):
                if nums[j] > nums[j+1]:
                    change = True
                    nums[j], nums[j+1] = nums[j+1], nums[j]
            
            if not change:
                return
            
    def quick_sort(self, nums):
        if len(nums) < 2:
            return nums
        
        # 快排
        pivot = nums[0]



        left = [x for x in nums if x < pivot]
        mid = [x for x in nums if x == pivot]
        right = [x for x in nums if x > pivot]

        return self.quick_sort(left)+mid+self.quick_sort(right)
    
    def quick_sort_1(self, nums):
        def qsort(l,r):
            if l >= r:
                return
            
            pivot = nums[l]

            i, j = l, r
            while(i < j):
                # 在i = j 找到pivot的坑位
                while i < j and nums[j] >= pivot:
                    j -= 1
                
                nums[i] = nums[j]

                while i < j and nums[i] <= pivot:
                    i += 1
                
                nums[j] = nums[i]
            
            nums[i] = pivot

            qsort(l,i-1)
            qsort(i+1,r)

        qsort(0, len(nums)-1)



        
# @lc code=end

