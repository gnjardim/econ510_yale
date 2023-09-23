# utility/production parameters
sigma = 2
alpha = 0.36
theta = 3^(-0.36)
delta = 0.2 / 3
beta = 1 / 1.05

# steady state
k_ss = (alpha * theta / (1/beta - 1 + delta))^(1 / (1 - alpha))

# create grid
N = 1000
k_grid = LinRange(0.5*k_ss, 1.5*k_ss, N)

# define utility function
function u(c)
    return (c .^ (1 - sigma) - 1) ./ (1 - sigma)
end


# Value Function Iteration with Grid Search
function vfi(tol = 1e-2, max_iter = 10000)

    # numerical parameters
    iter = 0
    diff = 1e10

    # setting up vectors
    V_old = zeros(N)
    V_new = zeros(N)
    V = zeros(N)
    policy_function = zeros(N)

    while (diff > tol && iter < max_iter)
        
        for i in 1:N
            k = k_grid[i]
            
            y = theta * (k ^ alpha) + (1 - delta) * k

            for j in 1:N
                k_prime = k_grid[j]
                
                if (y < k_prime)
                    V[j] = -Inf
                else
                    V[j] = u(y - k_prime) + beta * V_old[j]
                end
            end

            # find max for each k
            max_v = findmax(V)
            
            # save max value and index
            V_new[i] = max_v[1]
            index = max_v[2]
            
            # get policy function
            policy_function[i] = k_grid[index]
        end

        # For next step
        diff = maximum(abs.(V_new - V_old))
        V_old = copy(V_new)
        iter = iter + 1

        println(iter, ": ", diff)
    end
        
    return V_new, policy_function
end

# solve problem
sol = vfi()

# compute steady state using policy function
policy_k = sol[2]
k_ss_policy = k_grid[findmax(-abs.(policy_k - k_grid))[2]]
println("Analytical k_ss: ", k_ss, "\n", "Numerical k_ss: ", k_ss_policy)


# plots ----------------------------------------------------------
using Plots, LaTeXStrings

p1 = plot(k_grid, sol[1], xlabel = L"$k$", ylabel = L"$V(k)$", label = "")
p2 = plot(k_grid, sol[2], xlabel = L"$k$", ylabel = L"$k^{'} = g(k)$", label = "")
plot!(k_grid, k_grid, label = L"$y=x$") 

plot(p1, p2, layout = (2,1), plot_title = "Value Function Iteration")
annotate!(0.85, -15, ("Analytical \$ \\bar{k} = 3.135 \$", 6))
annotate!(0.85, -15.5, ("Computed \$ \\bar{k} = 3.124 \$", 6))
savefig("output/hw_3_q3_vfi.pdf")


# compute the transition paths ------------------------------------
k1 = 0.8*k_ss
k2 = 1.2*k_ss

T_ss = 150
function compute_k_sequence(k_0, T = T_ss)

    # find closest k in grid
    k_old = k_grid[findmax(-abs.(k_0 .- k_grid))[2]]

    # initialize k's
    k_new = copy(k_old)
    k_seq = zeros(T)

    # iterate until convergence (steady state) - actually, iterate N steps so it shows convergence
    for i in 1:T
        
        # find k' = g(k)
        k_new = policy_k[findmax(-abs.(k_old .- k_grid))[2]]

        # add to sequence
        k_seq[i] = k_old

        # next step
        k_old = k_new
        diff = abs(k_new - k_old)
    end

    return k_seq
end

k1_seq = compute_k_sequence(k1)
k2_seq = compute_k_sequence(k2)

# plot
p_k1 = plot(1:T_ss, k1_seq, label = L"$k_t$", title =  L"$k_0 = 0.8 k_{ss}$")
p_k2 = plot(1:T_ss, k2_seq, label = L"$k_t$", title =  L"$k_0 = 1.2 k_{ss}$")

plot(p_k1, p_k2, layout = (2,1), ylims = [0, Inf])
savefig("output/hw_3_q3_kseq.pdf")
