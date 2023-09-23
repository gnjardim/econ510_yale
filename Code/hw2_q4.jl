# utility/production parameters
sigma = 2
alpha = 0.36
theta = 3^(-0.36)
delta = 0.2 / 3
beta = 1 / 1.05

# steady state
T = 150
k_ss = (alpha * theta / (1/beta - 1 + delta))^(1 / (1 - alpha))

# algorithm parameters
update = 0.999

# compute initial sequence
function initial_k(k_0)

    k = LinRange(k_0, k_ss, T+2)

    return Array(k)
    
end

# shooting algorithm
function shooting(initial_k_seq, tol = 1e-5, max_iter = 10000)
    
    iter = 0
    diff = 1e10

    r_k = zeros(T)

    k = copy(initial_k_seq)
    k_euler = copy(k)

    while (diff > tol && iter < max_iter)

        # Compute the implied sequence of consumption
        c = max.(0, theta .* (k[1:T+1] .^ alpha) .+ (1 - delta) .* k[1:T+1] .- k[2:T+2])

        # Update the sequence of capital using the Euler equation
        r_k = (((c[2:T+1]./ c[1:T]).^sigma)/beta) .- 1 .+ delta   # from the Euler equation, we have r_k
        k_euler[2:T+1] = (r_k ./ (theta*alpha)) .^ (1/(alpha - 1))   # from the Firm FOC

        # New sequence of capital (update it slowly)
        k_new = update .* k + (1 - update) .* k_euler

        # For next step
        diff = maximum(abs.(k_new - k))
        k = copy(k_new)
        iter = iter + 1
        
    end

    # Compute other sequences
    k = k[1:T]
    r_b = r_k .- delta
    w = (1 - alpha)*theta * k.^alpha   

    return k, r_k, r_b, w

end

# compute initial sequence for k_0 = 0.5*k_ss and k_0 = 1.5*k_ss
k1 = initial_k(0.5*k_ss)
k2 = initial_k(1.5*k_ss)

# solve problem
sol_1 = shooting(k1)
sol_2 = shooting(k2)

# plots ----------------------------------------------------------
using Plots, LaTeXStrings

# for k1
p1_k1 = plot(1:T, sol_1[1], label = L"$k_t$")
p2_k1 = plot(1:T, sol_1[2], label = L"$r^{k}_t$")
p3_k1 = plot(1:T, sol_1[3], label = L"$r^{b}_t$")
p4_k1 = plot(1:T, sol_1[4], label = L"$w_t$")

plot(p1_k1, p2_k1, p3_k1, p4_k1, layout = (2,2), plot_title = L"$k_0 = 0.5 k_{ss}$", ylims = [0, Inf])
savefig("output/hw_2_q4_k1.pdf")

# for k2
p1_k2 = plot(1:T, sol_2[1], label = L"$k_t$")
p2_k2 = plot(1:T, sol_2[2], label = L"$r^{k}_t$")
p3_k2 = plot(1:T, sol_2[3], label = L"$r^{b}_t$")
p4_k2 = plot(1:T, sol_2[4], label = L"$w_t$")

plot(p1_k2, p2_k2, p3_k2, p4_k2, layout = (2,2), plot_title = L"$k_0 = 1.5 k_{ss}$", ylims = [0, Inf])
savefig("output/hw_2_q4_k2.pdf")

