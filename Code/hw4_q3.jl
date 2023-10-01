using Interpolations, Plots, LaTeXStrings

# utility/production parameters
mu = 0.5
alpha = 0.36
theta = 3^(-0.36)
delta = 0.2 / 3
beta = 1 / 1.05

# steady state
k_ss = (alpha * theta / (1/beta - 1 + delta))^(1 / (1 - alpha))
c_ss = theta * k_ss ^ alpha - delta * k_ss

# create grid
N = 50
K_grid = LinRange(0.5*k_ss, 1.5*k_ss, N)

# Recursive Equilibrium
function recursive_eq(tol = 1e-5, max_iter = 10000, update = 0.9)

    # numerical parameters
    iter = 0
    diffA = 1e10
    diffB = 1e10
    diff_pfi = 1e10

    # setting up vectors
    Ga_old = mu * repeat(K_grid, 1, N) + (1 - mu) * repeat(K_grid', N, 1)
    Ga_new = ones(N, N)
    Gb_old = mu * repeat(K_grid, 1, N) + (1 - mu) * repeat(K_grid', N, 1)
    Gb_new = ones(N, N)

    # define K
    K = mu * repeat(K_grid, 1, N) + (1 - mu) * repeat(K_grid', N, 1)

    # define R(K) and W(K) 
    R = (alpha * theta * (K.^(alpha - 1))) .+ 1 .- delta
    W = (1 - alpha) * theta * (K.^(alpha))

    # setting up t+1 vectors
    R_prime = ones(N, N)
    Ka_prime = ones(N, N)
    Kb_prime = ones(N, N)
    K_prime = ones(N, N)
    
    policy_old = ones(N, N, N) * c_ss
    policy_new = ones(N, N, N)
    policy_k = ones(N, N, N)

    # income
    Y = repeat(W, 1, 1, N) + repeat(R, 1, 1, N) .* repeat(reshape(K_grid, 1, 1, N), N, N, 1)
    
    while (diffA > tol && diffB > tol && iter < max_iter)

        Ka_prime = Ga_old
        Kb_prime = Gb_old
        K_prime = mu .* Ka_prime + (1 - mu) .* Kb_prime

        R_prime = (alpha * theta * (K_prime.^(alpha - 1))) .+ 1 .- delta
        W_prime = (1 - alpha) * theta * (K_prime.^(alpha))

        # Policy Function Iteration
        while (diff_pfi > tol)

            # compute k'
            k_prime = Y - policy_old

            # compute c', use interpolation and evaluate
            c_prime_interp = linear_interpolation((K_grid, K_grid, K_grid), policy_old, extrapolation_bc = Line())
            c_prime = [c_prime_interp(repeat(Ka_prime, 1, 1, N)[i, j, l], repeat(Kb_prime, 1, 1, N)[i, j, l], k_prime[i, j, l]) for i in 1:N, j in 1:N, l in 1:N]
            
            policy_new = (c_prime ./ beta) ./ repeat(R_prime, 1, 1, N)
            policy_new = min.(max.(1e-10, policy_new), Y)

            # For next step
            diff_pfi = maximum(abs.(policy_new - policy_old))
            policy_old = update * policy_old + (1 - update) * policy_new

        end

        # compute policy function for k 
        policy_k = Y - policy_new

        # Update our guess on the policy function
        for i in 1:N
            for j in 1:N
                Ga_new[i, j] = policy_k[i, j, i]
                Gb_new[i, j] = policy_k[i, j, j]
            end
        end

        # For next step
        diffA = maximum(abs.(Ga_new - Ga_old))
        diffB = maximum(abs.(Gb_new - Gb_old))
        Ga_old = update * Ga_old + (1 - update) * Ga_new
        Gb_old = update * Gb_old + (1 - update) * Gb_new

        iter = iter + 1
        println(iter, ": ", max(diffA, diffB))
    end
        
    return Ga_new, Gb_new, policy_k
end

### solve problem
eq = recursive_eq()

### transition path
T = 200
policy_k = eq[3]
policy_k_interp = linear_interpolation((K_grid, K_grid, K_grid), policy_k, extrapolation_bc = Line())

Ka_seq = zeros(T)
Kb_seq = zeros(T)

Ka_seq[1] = 0.5*k_ss
Kb_seq[1] = 1.2*k_ss

for t in 2:200
    Ka_seq[t] = policy_k_interp(Ka_seq[t-1], Kb_seq[t-1], Ka_seq[t-1])
    Kb_seq[t] = policy_k_interp(Ka_seq[t-1], Kb_seq[t-1], Kb_seq[t-1])
end

p_Ka = plot(1:T, Ka_seq, label = L"$K_A$")
p_Kb = plot(1:T, Kb_seq, label = L"$K_B$")

plot(p_Ka, p_Kb, layout = (2,1))
savefig("output/hw_4_q3.pdf")
