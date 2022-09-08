#include <algorithm>
#include <vector>
#include <set>
#include <assert.h>
#include <string>
#include <iostream>
#include <sstream>
#include <iterator>
#include <unordered_map>
#include <functional>
#include <numeric>
#include <math.h>

void printvec(const std::vector<int>& my_vector);

// Helper function: factorize a number (big_number) into sorted prime factors (factors_result)
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

// DefaultMapper's algorithm
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

// Helper function: divide a / b elementwise, return float
std::vector<float> divide(std::vector<int> a, std::vector<int> b)
{
    assert(a.size() == b.size());
    std::vector<float> result;
    for (int i = 0; i < a.size(); i++)
    {
        result.push_back((float) a[i] / (float) b[i]);
    }
    return result;
}


// This is not a good algorithm. Dynamic programming here is equivalent to brute force + memorization
std::vector<int> brute_force(int number, std::vector<int> launch_domain)
{
    // number can be regarded as #nodes
    int dim = launch_domain.size();
    std::vector<int> result;
    result.resize(dim, 1);
    if (number == 1)
    {
        return result;
    }

    // factorize workload constant into prime_nums (sorted from smallest to largest)
    std::vector<int> prime_nums;
    generate_prime_factor(number, prime_nums);
    // number = p_1 * p_2 * ... * p_m
    // number = o_1 * o_2 * ... * o_{dim}
    // There are dim^{m} different ways to map p_i (i<=m) to o_j (j<=dim)

    // Use <long long int> to represent the mapping state
    // Each digit is a base-{dim} number, and it has at most m digits
    // The first p_i has weight 1={base}^0, the next p_i has weight {base}^1, the next has {base}^2...
    
    // state_o_vec[k][state]: after placing first k primes (mapping recorded in state), the vector of o_i
    // vector<int>: o_1, o_2, ..., o_n
    std::vector<std::unordered_map<long long int, std::vector<int>>> state_o_vec;

    // Before choose any mapping, all o_{i} should be initialized to 1
    std::vector<int> vec_init(result);
    std::unordered_map<long long int, std::vector<int>> state_o_vec_0({{0, vec_init}});
    state_o_vec.push_back(state_o_vec_0);

    // For step k (selecting the p_k's mapping):
    // Enumerate all possible state \in { state | state_of_first_{k-1}_primes' mapping choices } 
    // Enumerate all possible mapping choices j for p_k (j can choose any index from 0 to dim)
    // state_o_vec[k][state \union p_k] =  state_w[k-1][state].update(j, multiplied by p_k)

    for (int i = 0; i < prime_nums.size(); i++) // Each step decides the mapping for p_i
    {
        std::unordered_map<long long int, std::vector<int>> state_o_vec_i;
        for (const auto item: state_o_vec[i])
        {
            const long long int pre_state = item.first;
            std::vector<int> pre_o_vec = item.second;
            for (int j = 0; j < dim; j++) // for each prime i, enumerate every o_j to be mapped
            {
                long long int new_state = pre_state + j * pow(dim, i);
                std::vector<int> new_o_vec(pre_o_vec);
                new_o_vec[j] *= prime_nums[i];
                assert(state_o_vec_i.count(new_state) == 0); // must be unique
                state_o_vec_i.insert({new_state, new_o_vec});
            }
        }
        state_o_vec.push_back(state_o_vec_i);
    }

    // minimize maximum difference: iterate each state in state_max[m][*]
    // Compute the workload vector w_i: w_i = L_i / o_i
    // find the smallest (max_element(w_i) - min_element(w_i))
    std::unordered_map<long long int, std::vector<int>> final_states = state_o_vec[state_o_vec.size()-1];
    float minimal_diff = INT32_MAX;
    for (const auto& item : final_states)
    {
        std::vector<int> o_vec = item.second;
        std::vector<float> w_vec = divide(launch_domain, o_vec);
        float cur = (*std::max_element(w_vec.begin(), w_vec.end())) - (*std::min_element(w_vec.begin(), w_vec.end()));
        if (cur < minimal_diff)
        {
            minimal_diff = cur;
            result = o_vec;
        }
    }
    return result;
    // Complexity analysis: suppose C=number, dim=launch_domain.size()
    // #State: #choices for mapping C's prime numbers into dim positions
    // #C's prime numbers: O(log(C, base=2))
    // #State: O({dim}^{log(C)})
    // Iterative picking prime: O(dim * {dim}^{log(C)}) = O({dim}^{log(C)+1})
    // Last For-loop: picking maximum/minimum among {dim} numbers for each state: O({dim}^{log(C)+1})
    // Complexity: O({dim}^{log(C)})
}


// quick pow for int
inline int binpow(int a, int b)
{
    int res = 1;
    while (b > 0)
    {
        if (b & 1)
        {
            res = res * a;
        }
        a = a * a;
        b >>= 1;
    }
    return res;
}

// 
std::vector<int> brute_force2(int number, std::vector<int> launch_domain)
{

}


// bool is_feasible(int min_w, int max_w, int number, const std::vector<int>& launch_domain)
// {
//     for (int i = 0; i < )
// }

std::vector<int> precise_enumerate(int number, std::vector<int> launch_domain)
{
    // number can be regarded as #nodes
    int dim = launch_domain.size();
    std::vector<int> result;
    result.resize(dim, 1);
    if (number == 1)
    {
        return result;
    }
    // Get all the factors for {number}
    std::vector<int> factors;


    return {};
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
    printvec(greedy(1024, std::vector<int>{2, 2, 8}));
    printvec(brute_force(1024, std::vector<int>{2, 2, 8}));
    printvec(brute_force2(1024, std::vector<int>{2, 2, 8}));
    // printvec(sliding_window(8, std::vector<int>{2, 2, 8}));
    return 0;
}
