#
# @lc app=leetcode.cn id=200 lang=python3
#
# [200] 岛屿数量
#
from typing import List
# @lc code=start
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        def changeIsland(grid, i, j):
            # 边界判断：超出网格 或 当前是水(0)，直接返回
            if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == "0":
                return
            
            # 把当前陆地变成水
            grid[i][j] = "0"
            
            # 递归淹没 下左右 的连通陆地
            changeIsland(grid, i - 1, j)  # 下
            changeIsland(grid, i + 1, j)  # 下
            changeIsland(grid, i, j - 1)  # 左
            changeIsland(grid, i, j + 1)  # 右
        
        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "0":
                    continue
                else:
                    # 找到岛屿，加入计数，然后将其转化为0
                    res += 1
                    changeIsland(grid, i, j)

        return res
        
# @lc code=end

