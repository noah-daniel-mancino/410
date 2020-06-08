using LinearAlgebra
using SparseArrays
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
    i = dimx * (bidx - 1) + tidx
    if i <= R
        for j = 1:C
            y[i] += (x[j] * A[i,j]) + (b[j] / C)
        end
    end
    return nothing
end

function my_forward_Euler(Δt, t1, tf, y, Δx, myexact_fun)

	N = Integer((tf - t1)/Δt)  # total number of points is N + 1
    M = length(y) + 2
	Y = Matrix{Float64}(undef,M,N+1)
    Y[:,1] = [0;y[:];0]
    t = t1:Δt:tf

    A = Δt*(2/Δx^2)*(sparse(1:P-1,1:P-1,-2*ones(P-1), P-1, P-1) + sparse(2:P-1,1:P-2,ones(P-2),P-1,P-1) +
    sparse(1:P-2,2:P-1,ones(P-2),P-1,P-1))
    A = CuArray(A)
    
	Exact = Matrix{Float64}(undef,M,N+1)
    Exact[:,1] = [0;y[:];0]
    b = Array{Float64}(undef, P-1)
    empty = zeros(length(y))
    threads_per_x = 32
    num_blocks_x = cld(size(A)[1], threads_per_x)

    for n = 2:N+1
        @show P
        for k = 1:P-1
            b[k] = Δt*G(Δx*k, t[n-1])
        end
        b_d = CuArray(b) 
        y_d = CuArray(empty)
        x_d = CuArray(y)
        @cuda threads=threads_per_x blocks=num_blocks_x knl_gemv!(A, x_d, b_d, y_d)
        synchronize()
        y[:] .+= y_d
        @show y
        @show myexact_fun(t[n])
        Y[:,n] = [0;y[:];0]
        Exact[:,n] = myexact_fun(t[n])
    end

    return (t, Y, Exact)
end

function exact(t)
    y = Array{Float64}(undef, P+1)
    y[1] = 0
    for n = 2:P
        y[n] = exp(-2t)*sin(pi*(n-1)*Δx)
    end
    y[P+1] = 0
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
λ = 0.25
Δt = Δx^2 * λ
T = 1

P  = Integer((1-0)/Δx) # N+1 total nodes, N-1 interior nodes

y = Array{Float64}(undef, P-1)
for n = 2:P
    y[n-1] = f(Δx*(n-1))
end

(t, U, E) = my_forward_Euler(Δt, 0, T, y, Δx, exact)

@show error(exact(1), U[:,length(t)])