using SparseArrays
using LinearAlgebra
using Plots
include("./myForwBack_Euler.jl")
 
# Solve steady state and time-dependent heat equation in 1D
# u_t  = k u_xx + G(x,t) on 0 ≦ x ≦ 1
# u(x, 0) = f(x)
# u(0, t) = u(1, t) = 0

# Steady state: 0 = k u_xx + F(x), u(0) = u(1) = 0


function rhs(t, y)
    A = (2/Δx^2)*(sparse(1:N-1,1:N-1,-2*ones(N-1), N-1, N-1) + sparse(2:N-1,1:N-2,ones(N-2),N-1,N-1) +
    sparse(1:N-2,2:N-1,ones(N-2),N-1,N-1))

    y = y[2:N]

    b = Array{Float64}(undef, N-1)
    for n = 1:N-1
        b[n] = G(Δx*n, t)
    end

    y1 = Array{Float64}(undef, N-1)
    y1 = A*y
    y1 .= y1 + b

    return [0;y1;0]
end


function F(x)
    return k*pi^2*sin.(pi*x)
end

function G(x,t)
    return 2(pi^2 - 1)*exp(-2t)*sin.(pi*x)
end

function f(x)
    return sin.(pi*x)
end

function exact_steadystate(x)
    return sin.(pi*x)
end

function back_time_dependent_heat(k, Δx, Δt, T, my_source, my_initial)
    x = 0:Δx:1
    t = 0:Δt:T
    M = length(t) - 1#Integer(ceil((T-0)/Δt)) # M+1 total temporal nodes
    λ = Δt/Δx^2


    # A is N-1 by N-1

    A = (2/Δx^2)*(sparse(1:N-1,1:N-1,-2*ones(N-1), N-1, N-1) + sparse(2:N-1,1:N-2,ones(N-2),N-1,N-1) +
    sparse(1:N-2,2:N-1,ones(N-2),N-1,N-1))

    u = Array{Float64}(undef,N-1)
    u .= my_initial(x[2:N])  # setting the initial condition for interior nodes

    U = Array{Float64}(undef,N+1,M+1)  # whole numerical solution
    U[:,1] = my_initial(x[1:N+1])


    Id = Matrix{Float64}(I,N-1,N-1)

    # I thought backwards euler in this case was (Id - ΔtA)x + Δt*c = y(n-1).
    # I'm wrong but I really have no clue how. 
    for n = 1:M
        b = Δt * my_source(x[2:N],t[n])
        u[:] = (Id - Δt*A)\(u[:] + b)
        U[:,n+1] = [0;u;0]
    end

    return (x, t, U)
end

function exact(t)
    y = Array{Float64}(undef, N+1)
    y[1] = 0
    for n = 2:N
        y[n] = exp(-2t)*sin(pi*(n-1)*Δx)
    end
    y[N+1] = 0
    return y
end

function error(exact, numerical)
    sum = 0
    for n = 1:N+1
        sum += (numerical[n] - exact[n])^2 * Δx
    end
    return sum
end

# Please note that for this script to work k, Δx, Δy, T, and N must be defined 
# as global variables.

k = 2
Δx = 0.1
λ = 0.1
Δt = 0.1
T = 2

N  = Integer((1-0)/Δx) # N+1 total nodes, N-1 interior nodes

y = Array{Float64}(undef, N+1)
y[1] = 0
for n = 2:N
    y[n] = f(Δx*(n-1))
end
y[N+1] = 0


(t,U,E) = Euler.my_forward_Euler(Δt, 0, T, rhs, y, exact)
@show error(exact(2), U[:,length(t)])


(meh, meh, U) = back_time_dependent_heat(k,Δx,Δt,T,G,f)
@show error(exact(2), U[:,length(t)])
@show U[:,length(t)]
@show exact(2)