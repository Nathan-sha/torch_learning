#
# @lc app=leetcode.cn id=75 lang=python3
#
# [75] 颜色分类
#
from typing import List

# @lc code=start
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        left, right = 0, len(nums)-1
        # 分别是0和1的最新目标位置

        now = 0
        while(now < right+1):
            if nums[now] == 0:
                nums[now], nums[left] = nums[left], nums[now]
                left += 1
                now += 1
            elif nums[now] == 2:
                nums[now], nums[right] = nums[right], nums[now]
                right -= 1
            else:
                now += 1
                
        



# @lc code=end

