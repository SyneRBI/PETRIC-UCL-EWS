
import numpy as np 
import math 

def compute_number_of_subsets(views):
    """
    At least 8 views per subset
    
    """
    
    """
    n_subs = math.floor(views / 8)
    # views has to be divisible by n_subs

    while views % n_subs != 0:
        n_subs -= 1

        if n_subs < 1: break 
    return int(max(n_subs, 1))
    """
    num_divisors = list(divisorGenerator(views))
    subset_size = [int(views / div) for div in num_divisors]
    num_primes = [len(prime_factors(div)) for div in num_divisors]

    combined = list(zip(num_divisors, subset_size, num_primes))
    combined = [(x,y,z) for (x,y,z) in combined if y >= 8][::-1]
    combined_filtered = [(x,y,z) for (x,y,z) in combined if y <= 0.2*views]
    
    if len(combined_filtered) < 2:
        subset_number =  combined[0][0] 
    else:
        num_div, subset_size, num_primes = list(zip(*combined_filtered))

        amax = np.argmax(num_primes)

        subset_number = num_div[amax]

    return subset_number

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

def divisorGenerator(n):
    large_divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(int(n / i))
    
    for divisor in reversed(large_divisors):
        yield divisor

if __name__ == "__main__":

    for num in [50, 252, 128]:
        print("Test for ", num, " -> ", compute_number_of_subsets(num), " subsets")

            

    """

    n_subs = compute_number_of_subsets(50)
    print(50, n_subs, prime_factors(50), list(divisorGenerator(50)))

    n_subs = compute_number_of_subsets(252)
    print(252, n_subs, prime_factors(252), list(divisorGenerator(252)))

    n_subs = compute_number_of_subsets(128)
    print(128, n_subs, prime_factors(128), list(divisorGenerator(128)))

    """