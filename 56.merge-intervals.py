#
# @lc app=leetcode id=56 lang=python3
#
# [56] Merge Intervals
#
from typing import List

# @lc code=start
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        res = [[]]
        for i in intervals:
            if len(res) == 0:
                res.append(i)

            for j in res:
                temp = self.ifOverlapping(i,j)
                if len(temp == 1):

                for q in temp:
                    res.append(q)

        return res
    
    def ifOverlapping(self, oldInterval: List[int], newInterval: List[int]):
        # 如果不重叠 返回输入 如果重叠 返回修正后的 结果都为List[List[int]]
        res = [[]]
        oldL, oldR = oldInterval[0], oldInterval[1]
        newL, newR = newInterval[0], newInterval[1]

        if newR < oldL or newL > oldR:
            res.append(oldInterval)
            res.append(newInterval)
            return res
        else:
            left = min(oldL, newL)
            right = max(oldR, newR)
            temp = [left,right]
            res.append(temp)
            return res


# @lc code=end


class Solution:
    def merge(self, input: List[List[int]]) -> List[List[int]]:
        
        pass

