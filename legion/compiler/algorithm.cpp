#include <algorithm>
#include <vector>
#include <assert.h>
#include <string>
#include <iostream>
#include <sstream>
#include <iterator>

// This function factorize a number (big_number) into sorted prime factors (factors_result)
void generate_prime_factor(int big_number, std::vector<int>& factors_result)
{
    auto generate_factors = [&](int factor)
    {
        while (big_number % factor == 0)
        {
            factors_result.push_back(factor);
            big_number /= factor;
        }
    };
    // Fundamental theorem of arithmetic time!
    const unsigned num_primes = 32;
    const int primes[num_primes] = { 2, 3, 5, 7, 11, 13, 17, 19, 
                                    23, 29, 31, 37, 41, 43, 47, 53,
                                    59, 61, 67, 71, 73, 79, 83, 89,
                                    97, 101, 103, 107, 109, 113, 127, 131 };
    // Increase the size of the prime number table if you ever hit this
    assert(big_number <= (primes[num_primes-1] * primes[num_primes-1]));
    for (int i = 0; i < num_primes; i++)
    {
        if (primes[i] * primes[i] > big_number) // There is at most 1 prime whose index >= i
            break;
        generate_factors(primes[i]);
        if (big_number == 1)
            break;
    }
    if (big_number > 1) // E.g., we need to put 137 into the result for 2 * 137
        factors_result.push_back(big_number);
}

std::vector<int> greedy(int number, std::vector<int> launch_domain)
{
    int dim = launch_domain.size();
    std::vector<int> result;
    result.resize(dim, 1);
    if (number == 1)
    {
        return result;
    }

    // factorize number into prime_nums (sorted from smallest to largest)
    std::vector<int> prime_nums;
    generate_prime_factor(number, prime_nums);
    // Assign prime nums onto the dimensions
    // from the largest primes down to the smallest, in a greedy approach
    std::vector<double> domain_vec;
    for (int i = 0; i < dim; i++)
        domain_vec.push_back((double) launch_domain[i]); // integer to double

    // from the largest primes down to the smallest
    for (int idx = prime_nums.size() - 1; idx >= 0; idx--)
    {
        // Find the dimension with the biggest domain_vec
        int next_dim = std::max_element(domain_vec.begin(), domain_vec.end()) - domain_vec.begin();

        int next_prime = prime_nums[idx];

        result[next_dim] *= next_prime;
        domain_vec[next_dim] /= next_prime;
    }
    return result;
}

std::vector<int> brute_force(int number, std::vector<int> launch_domain)
{
    int dim = launch_domain.size();
    std::vector<int> result;
    result.resize(dim, 1);
    if (number == 1)
    {
        return result;
    }

    // factorize number into prime_nums (sorted from smallest to largest)
    std::vector<int> prime_nums;
    generate_prime_factor(number, prime_nums);
}

std::vector<int> sliding_window(int number, std::vector<int> launch_domain)
{

}

inline std::string vec2str(const std::vector<int>& my_vector)
{
    std::stringstream result;
    std::copy(my_vector.begin(), my_vector.end(), std::ostream_iterator<int>(result, " "));
    return result.str();
}

void printvec(const std::vector<int>& my_vector)
{
    std::cout << vec2str(my_vector) << std::endl;
}

int main()
{
    printvec(greedy(8, std::vector<int>{2, 2, 8}));
    return 0;
}
