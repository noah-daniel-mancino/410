using LinearAlgebra
using SparseArrays
using Plots
using CuArrays
using CUDAnative
using CUDAdrv

function G(x,t)
    return 2(pi^2 - 1)*exp(-2t)*sin.(pi*x)
end

function knl_gemv!(A, x, b, y)
    (R, C) = size(A)
    bidx = blockIdx().x
    tidx = threadIdx().x
    dimx = blockDim().x
	M = length(x)
	K = length(y)

    bidy = blockIdx().y
    tidy = threadIdx().y
    dimy = blockDim().y
    i = dimx * (bidx - 1) + tidx
    j = dimy * (bidy - 1) + tidy
    if i <= R && j <= K
        y[i] += (x[j] * A[i,j]) + (b[j] / C)
    end
    return nothing
end

function my_forward_Euler(Δt, t1, tf, y, Δx, myexact_fun)

	N = Integer((tf - t1)/Δt)  # total number of points is N + 1
	M = length(y)
	Y = Matrix{Float64}(undef, M, N+1)
    Y[:,1] = y[:]
    t = t1:Δt:tf

    A = (2/Δx^2)*(sparse(1:N-1,1:N-1,-2*ones(N-1), N-1, N-1) + sparse(2:N-1,1:N-2,ones(N-2),N-1,N-1) +
    sparse(1:N-2,2:N-1,ones(N-2),N-1,N-1))
    A = CuArray(A)
    
	Exact = Matrix{Float64}(undef,M,N+1)
    Exact[:,1] = y[:]
    b = Array{Float64}(undef, N-1)
    empty = zeros(length(y))
    threads_per_x = 32
    threads_per_y = 32
    thd_tup = (threads_per_x, threads_per_y)
    num_blocks_x = cld(size(A)[0], threads_per_x)
    num_blocks_y = cld(size(A)[1], threads_per_y)
    blocks_tup = (num_blocks_x, num_blocks_y)

    for n = M:N+1
        for k = 1:N-1
            b[n] = G(Δx*k, t[n-1])
        end
        b_d = CuArray(b) 
        y_d = CuArray(empty)
        x_d = CuArray(y)
        @cuda threads=thd_tup blocks=blocks_tup knl_gemv!(A, x_d, b_d, y_d)
        synchronize()
        y[:] .= [0;y_d;0]
        Y[:,n] = y[:]
        Exact[:,n] = myexact_fun(t[n])
    end

    return (t, Y, Exact)
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

function f(x)
    return sin.(pi*x)
end
k = 2
Δx = 0.1
λ = 0.1
Δt = 0.001
T = .1

N  = Integer((1-0)/Δx) # N+1 total nodes, N-1 interior nodes

y = Array{Float64}(undef, N+1)
y[1] = 0
for n = 2:N
    y[n] = f(Δx*(n-1))
end
y[N+1] = 0

(t, U, E) = my_forward_Euler(Δt, 0, T, y, Δx, exact)

@show U[:1,]
@show E[:1,]