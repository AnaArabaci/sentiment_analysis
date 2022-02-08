# For line-by-line memory usage

# Put code in a function
# Wrap the function in a decorator

#import line_profiler
#profile = line_profiler.LineProfiler()
#from memory_profiler import profile

@profile
def my_func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    return a

if __name__ == '__main__':
    my_func()

# Execute the code passing the option -m memory_profiler to the python interpreter
# to load the memory_profiler module and print to stdout the line-by-line analysis

# $ python -m memory_profiler example.py