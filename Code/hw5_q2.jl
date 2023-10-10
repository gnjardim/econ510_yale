# utility/production parameters
alpha = 0.3
delta = 0.07
beta = 0.99
epsilon = 0.05
z_grid = [1 - epsilon, 1 + epsilon]
nz = 2

# steady state
k_ss_high = (alpha * z_grid[2] / (1/beta - 1 + delta))^(1 / (1 - alpha))
k_ss_low = (alpha * z_grid[1] / (1/beta - 1 + delta))^(1 / (1 - alpha))

# create grid
nk = 1000
k_grid = LinRange(0.8*k_ss_low, 1.5*k_ss_high, nk)

# define utility function
function u(c)
    if c > 0
        return log(c)
    else
        return -Inf
    end
end

# Value Function Iteration with Grid Search
function vfi(p = 0.5, tol = 1e-2, max_iter = 10000)

    # numerical parameters
    iter = 0
    diff = 1e10

    # setting up vectors
    V_old = zeros(nz, nk)
    V_new = zeros(nz, nk)
    policy_function = zeros(nz, nk)
    U = zeros(nz, nk, nk)

    # matrix of probabilities
    P = [p 1-p; 1-p p] 

    for i = 1:nz
        for j = 1:nk
            C = z_grid[i] * k_grid[j]^alpha + (1 - delta) * k_grid[j] .- k_grid
            U[i,j,:] = u.(C)
        end
    end

    while (diff > tol && iter < max_iter)
        
        # E[V(z', k')]
        EV = repeat(reshape(P*V_old, nz, 1, nk), 1, nk, 1)

        # find max for each z, k (over k')
        max_v = findmax(U + beta*EV, dims = 3)
        
        # save max value and index
        V_new = max_v[1][:, :, 1]
        index = mapslices(argmax, U + beta*EV, dims = 3)[:, :, 1]
        
        # get policy function
        for i in 1:nz
            for j in 1:nk
                policy_function[i, j] = k_grid[index[i, j]]
            end
        end

        # For next step
        diff = maximum(abs.(V_new - V_old))
        V_old = copy(V_new)
        iter = iter + 1

        if rem(iter, 100) == 0
            println(iter, ": ", diff)
        end
    end
        
    return V_new, policy_function
end

# solve problem
sol = vfi()
policy_k = sol[2]


# change matrix of probabilities ---------------------------------
sol_08 = vfi(0.8)
policy_k_08 = sol_08[2]


# plots ----------------------------------------------------------
using Plots, LaTeXStrings

p1 = plot(k_grid, policy_k[1, :], xlabel = L"$k$", ylabel = L"$k^{'} = g(k)$", label = L"z_L", title = "p = 0.5")
plot!(k_grid, policy_k[2, :], xlabel = L"$k$", ylabel = L"$k^{'} = g(k)$", label = L"z_H")
plot!(k_grid, k_grid, label = L"$y=x$", linestyle = :dash) 


p2 = plot(k_grid, policy_k_08[1, :], xlabel = L"$k$", ylabel = L"$k^{'} = g(k)$", label = L"z_L", title = "p = 0.8")
plot!(k_grid, policy_k_08[2, :], xlabel = L"$k$", ylabel = L"$k^{'} = g(k)$", label = L"z_H")
plot!(k_grid, k_grid, label = L"$y=x$", linestyle = :dash) 

plot(p1, p2, layout = (1,2), plot_title = "Policy Functions")
savefig("output/hw_5_q2.pdf")