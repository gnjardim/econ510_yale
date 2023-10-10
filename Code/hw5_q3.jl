# utility/production parameters
r = 0.02
beta = 0.96
epsilon = 0.1
z_grid = [1 - epsilon, 1 + epsilon]
nz = 2
bbar = -0.1

# max and min k
kmax = 0.3 * (1 + r)/r * (z_grid[2] - z_grid[1])
kmin = bbar

# create grid
nk = 1000
k_grid = LinRange(kmin, kmax, nk)

# define utility function
function u(c)
    if c > 0
        return log(c)
    else
        return -Inf
    end
end

# Value Function Iteration with Grid Search
function vfi(p = 0.8, tol = 1e-2, max_iter = 10000)

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
            C = z_grid[i] + (1 + r) * k_grid[j] .- k_grid
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


# plots ----------------------------------------------------------
using Plots, LaTeXStrings

p1 = plot(k_grid, policy_k[1, :], xlabel = L"$k$", ylabel = L"$k^{'} = g(k)$", label = L"z_L", title = L"p = 0.8, ϵ = 0.1, \underbar{b} = -0.1")
plot!(k_grid, policy_k[2, :], xlabel = L"$k$", ylabel = L"$k^{'} = g(k)$", label = L"z_H")
plot!(k_grid, k_grid, label = L"$y=x$", linestyle = :dash) 
savefig("output/hw_5_q3.pdf")


# change matrix of probabilities ---------------------------------
sol_03 = vfi(0.3)
policy_k_03 = sol_03[2]

p_prob = plot(k_grid, policy_k_03[1, :], xlabel = L"$k$", ylabel = L"$k^{'} = g(k)$", label = L"z_L", title = L"p = 0.3")
plot!(k_grid, policy_k_03[2, :], xlabel = L"$k$", ylabel = L"$k^{'} = g(k)$", label = L"z_H")
plot!(k_grid, k_grid, label = L"$y=x$", linestyle = :dash) 

plot(p1, p_prob, layout = (1,2), plot_title = "Varying p")
savefig("output/hw_5_q3_prob.pdf")


# change magnitude of shocks -------------------------------------
epsilon = 0.5
z_grid = [1 - epsilon, 1 + epsilon]

# max k
kmax = 0.3 * (1 + r)/r * (z_grid[2] - z_grid[1])

# create grid
k_grid = LinRange(kmin, kmax, nk)

sol_eps05 = vfi()
policy_k_eps05 = sol_eps05[2]

p_eps = plot(k_grid, policy_k_eps05[1, :], xlabel = L"$k$", ylabel = L"$k^{'} = g(k)$", label = L"z_L", title = L"ϵ = 0.5")
plot!(k_grid, policy_k_eps05[2, :], xlabel = L"$k$", ylabel = L"$k^{'} = g(k)$", label = L"z_H")
plot!(k_grid, k_grid, label = L"$y=x$", linestyle = :dash) 

plot(p1, p_eps, layout = (1,2), plot_title = "Varying ϵ")
savefig("output/hw_5_q3_epsilon.pdf")


# change borrowing limit -------------------------------------
bbar = -1
epsilon = 0.1

z_grid = [1 - epsilon, 1 + epsilon]

# max k
kmax = 0.3 * (1 + r)/r * (z_grid[2] - z_grid[1])

# min k
kmin = bbar

# create grid
k_grid = LinRange(kmin, kmax, nk)

sol_bbar05 = vfi()
policy_k_bbar05 = sol_bbar05[2]

p_bbar = plot(k_grid, policy_k_bbar05[1, :], xlabel = L"$k$", ylabel = L"$k^{'} = g(k)$", label = L"z_L", title = L"\underbar{b} = -0.5")
plot!(k_grid, policy_k_bbar05[2, :], xlabel = L"$k$", ylabel = L"$k^{'} = g(k)$", label = L"z_H")
plot!(k_grid, k_grid, label = L"$y=x$", linestyle = :dash) 

plot(p1, p_bbar, layout = (1,2), plot_title = "Varying b")
savefig("output/hw_5_q3_bbar.pdf")


