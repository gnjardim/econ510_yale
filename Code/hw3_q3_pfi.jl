using Interpolations

# utility/production parameters
sigma = 2
alpha = 0.36
theta = 3^(-0.36)
delta = 0.2 / 3
beta = 1 / 1.05

# steady state
k_ss = (alpha * theta / (1/beta - 1 + delta))^(1 / (1 - alpha))
c_ss = theta * k_ss ^ alpha - delta * k_ss

# create grid
N = 1000
k_grid = LinRange(0.5*k_ss, 1.5*k_ss, N)

# Policy Function Iteration with Grid Search
function pfi(tol = 1e-5, max_iter = 10000)

    # numerical parameters
    iter = 0
    diff = 1e10

    # setting up vectors
    policy_old = ones(N) * c_ss # can't be zero!
    policy_new = zeros(N)
    
    while (diff > tol && iter < max_iter)
        
        for i in 1:N
            # compute k and k'
            k = k_grid[i]
            k_prime = theta * k^alpha + (1 - delta) * k - policy_old[i]

            # compute c', use interpolation and evaluate at k_prime
            c_prime_interp = linear_interpolation(k_grid, policy_old, extrapolation_bc = Line())
            c_prime = c_prime_interp(k_prime)
            
            policy_new[i] = ((c_prime^sigma / beta) / (theta * alpha * k_prime^(alpha - 1) + 1 - delta)) ^ (1/sigma)
            

        end

        # For next step
        diff = maximum(abs.(policy_new - policy_old))
        policy_old = copy(policy_new)
        iter = iter + 1

        println(iter, ": ", diff)
    end
        
    policy_c = policy_new
    policy_k = theta * k_grid .^ alpha + (1 - delta) * k_grid - policy_c

    return policy_k, policy_c
end

# solve problem
policy = pfi()
policy_k = policy[1]
policy_c = policy[2]

# compute steady state using policy function
k_ss_policy = k_grid[findmax(-abs.(policy_k - k_grid))[2]]
println("Analytical k_ss: ", k_ss, "\n", "Numerical k_ss: ", k_ss_policy)


# plots ----------------------------------------------------------
using Plots, LaTeXStrings

p1 = plot(k_grid, policy_c, xlabel = L"$k$", ylabel = L"$c = h(k)$", label = "")

p2 = plot(k_grid, policy_k, xlabel = L"$k$", ylabel = L"$k^{'} = g(k)$", label = "")
plot!(k_grid, k_grid, label = L"$y=x$") 

plot(p1, p2, layout = (2,1), plot_title = "Policy Function Iteration")
annotate!(0.85, -15, ("Analytical \$ \\bar{k} = 3.135 \$", 6))
annotate!(0.85, -15.5, ("Computed \$ \\bar{k} = 3.137 \$", 6))

savefig("output/hw_3_q3_pfi.pdf")