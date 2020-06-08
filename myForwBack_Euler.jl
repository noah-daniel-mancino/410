module Euler
using LinearAlgebra
using Plots

function my_forward_Euler(Δt, t1, tf, rhs, y, myexact_fun)

	N = Integer((tf - t1)/Δt)  # total number of points is N + 1
	M = length(y)
	Y = Matrix{Float64}(undef, M, N+1)
    Y[:,1] = y[:]
	t = t1:Δt:tf
    
	Exact = Matrix{Float64}(undef,M,N+1)
    Exact[:,1] = y[:]

    for n = M:N+1
        y[:] = y[:] + Δt*rhs(t[n-1], y[:])
        Y[:,n] = y[:]
        Exact[:,n] = myexact_fun(t[n])
    end

    return (t, Y, Exact)

end

end