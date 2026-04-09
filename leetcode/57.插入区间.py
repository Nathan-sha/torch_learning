#
# @lc app=leetcode.cn id=57 lang=python3
#
# [57] 插入区间
#
from typing import List
# @lc code=start
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        res = []
        if len(intervals) == 0:
            res.append(newInterval)
        for i in range(len(intervals)):
            if newInterval[0] > intervals[i][1]:
                if i == len(intervals) - 1:
                    res = intervals
                    res.append(newInterval)
                    break
                else:
                    continue
            elif newInterval[1] < intervals[i][0]:
                res = intervals
                res.insert(i, newInterval)
                break
            else:
                left = min(newInterval[0], intervals[i][0])
                while(i <= len(intervals)-1 and newInterval[1] >= intervals[i][0]):
                    temp = intervals.pop(i)
                right = max(newInterval[1], temp[1])
                res = intervals
                res.insert(i, [left, right])
                break

        return res    
        
# @lc code=end

