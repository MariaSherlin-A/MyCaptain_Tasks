def fibonacci(limit):
    fib_sequence = [0, 1]
    while True:
        next_fib = fib_sequence[-1] + fib_sequence[-2]
        if next_fib <= limit:
            fib_sequence.append(next_fib)
        else:
            break
    return fib_sequence

# Test
limit = int(input("Enter the limit for Fibonacci numbers: "))
fib_numbers = fibonacci(limit)
print("Fibonacci numbers up to", limit, ":", fib_numbers)
