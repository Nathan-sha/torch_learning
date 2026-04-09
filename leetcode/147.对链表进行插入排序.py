#
# @lc app=leetcode.cn id=147 lang=python3
#
# [147] 对链表进行插入排序
#
from typing import Optional
# @lc code=start
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def insertionSortList(self, head: Optional[ListNode]) -> Optional[ListNode]:

        dummy = ListNode(next=head)
        if(not head.next or not head):
            return head
        
        cur = head

        while(cur.next):
            pre = dummy
            # 从表头开始找插入位置


            while pre.next.val < cur.next.val and pre.next:
                pre = pre.next

            if pre == cur:
                cur = cur.next
            else:
                temp = cur.next
                cur.next = cur.next.next
                temp.next = pre.next
                pre.next = temp


        return dummy.next
                
            

# @lc code=end

