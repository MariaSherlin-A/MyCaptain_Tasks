def print_positive_numbers(lst):
    positive_numbers = [num for num in lst if num > 0]
    if positive_numbers:
        print("Output:", ", ".join(map(str, positive_numbers)))
    else:
        print("Output: No positive numbers found")

# Test cases
list1 = [12, -7, 5, 64, -14]
list2 = [12, 14, -95, 3]

print("Input: list1 =", list1)
print_positive_numbers(list1)

print("Input: list2 =", list2)
print_positive_numbers(list2)
