#
# @lc app=leetcode.cn id=207 lang=python3
#
# [207] 课程表
#
from typing import List

from collections import deque

# @lc code=start
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        table = {}
        # 存储该课程的前序课程，如没存，说明不需要前序课程
        q = deque()

        q.popleft()
        
        for i in range(len(prerequisites)):
            if prerequisites[i][0] in table:
                table[prerequisites[i][0]] += 1
            else:
                table[prerequisites[i][0]] = 1
        
    



        



# @lc code=end

