#
# @lc app=leetcode id=56 lang=python3
#
# [56] Merge Intervals
#
from typing import List

# @lc code=start
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        res = []
        for interval in intervals:
            if len(res) == 0:
                res.append(interval)
                continue

            for j in range(len(res)):
                if interval[1] < res[j][0]:
                    res.insert(j, interval)
                    break
                elif interval[0] > res[j][1]:
                    if j != len(res)-1:
                        continue
                    else:
                        res.append(interval)
                else:
                    left = min(interval[0], res[j][0])
                    while(j < len(res) and interval[1] >= res[j][0]):
                        temp = res.pop(j)
                    right = max(interval[1], temp[1])

                    res.insert(j, [left, right])
                    break                 

        return res


# @lc code=end
