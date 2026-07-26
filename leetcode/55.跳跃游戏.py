#
# @lc app=leetcode.cn id=55 lang=python3
#
# [55] 跳跃游戏
#

# @lc code=start
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        length = len(nums)
        # 同时记录最远的和当前遍历的
        MAX = 0
        for i in range(length):
            if MAX >= length-1:
                return True
            
            MAX = max(MAX, i + nums[i])
            if MAX <= i:
                return False
            
            


        
# @lc code=end

