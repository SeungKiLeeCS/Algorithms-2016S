#using quicksort

import sys
import matplotlib.pyplot as plt
import numpy as np
import time

list_data = []
with open(sys.argv[1]) as f:
    for line in f:
        line = line.strip()
        list_data.append(line)
list_data = map(int, list_data)

#quicksort -> the conquer part of the algorithm
def quicksort(input_list, start_index, end_index):
    if start_index < end_index:
        #partition the list of integers.
        pivot = partition(input_list, start_index, end_index)
        #recursive call on both sides of the pivot, that is excluding the pivot itself
        quicksort(input_list, start_index, pivot-1)
        quicksort(input_list, pivot+1, end_index)

    return input_list

#divide part of the algorithm
def partition(input_list, start_index, end_index):
    #declare variables required for sorting
    pivot = input_list[start_index]
    left = start_index + 1
    right = end_index
    sorted = False

    while not sorted:
        #break condition so that left index is crossed with right index
        #or if the value of left crosses the pivot value
        while left <= right and input_list[left] <= pivot:
            #increment left index
            left = left + 1
        #break the loop when right and left indexes cross
        #or if the right value crosses the pivot value
        while right >= left and input_list[right] >= pivot:
            right = right-1
        if right < left:
            sorted = True
        else:
            #swap places for left value and the right value cause they are not in order
            temp = input_list[left]
            input_list[left] = input_list[right]
            input_list[right] = temp
    #swap the value at start index with what's now at the right half. Then return right for the new pivot
    temp = input_list[start_index]
    input_list[start_index] = input_list[right]
    input_list[right] = temp
    return right

start_time = time.time()
quicksort(list_data, 0, len(list_data)-1)
elapsed_time = time.time() - start_time

print "Sorted List: " , list_data
print "Total Time Spent (Precise to Milliseconds): ", elapsed_time, " Seconds"

x = (0, len(list_data), 1000)
y = (0, elapsed_time, 0.001)

plt.plot(x,y)

plt.xlabel("Database Size")
plt.ylabel("Time Sorted (s)")
plt.title("Quicksort : 5000 elements")
plt.grid(True)
plt.savefig("part2_graph.png")
plt.show()
