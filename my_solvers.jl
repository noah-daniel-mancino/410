using LinearAlgebra
using SparseArrays

"""
	LUPsolve(a)

Compute and return LU factorization of square matrix a.

# Examples
'''
julia> A = rand(3,3)
julia> A = (L, U) = LUPsolve(a)
'''
"""
function computeLU(A)
	N = size(A)[1]

	Id = Matrix{Float64}(I, N, N)
	ell = copy(Id) 
	ell_inv = copy(Id)
	Atilde = copy(A)
	L = copy(Id)

	for k = 1:N-1 
		
		ell .= Id
		ell_inv .= Id
		for i = k+1:N 
			ell[i,k] = -Atilde[i,k] / Atilde[k,k] 
			ell_inv[i,k] = Atilde[i,k] / Atilde[k, k]
		end

		Atilde .= ell * Atilde
		L .= L * ell_inv

		end
		U = Atilde
	return (L, U)
end

"""
	computeLUP(A)

Compute and return the LUP factorization of a square matrix A.
function computeLUP(A)

# Examples
julia> A = rand(3,3)
julia> (L, U, P) = computeLUP(A)
julia> @assert P * A = L * U
"""

function computeLUP(A) 
	N = size(A)[1]
	B = copy(A)

	P = Matrix{Float64}(I, N, N)
	"""	
	To get the permutation matrix, we move through A column by column, finding
	the largest element below or at the pivot row. Then, we swap the row of
	the identity matrix at the pivot point with the row at the index of the
	largest element in the column.
	"""
	for k = 1:N-1 
		biggest_pivot = B[k,k]
		biggest_pivot_loc = k
		for i = k+1:N
			if max(biggest_pivot, B[i, k]) != biggest_pivot
				biggest_pivot = B[i, k]
				biggest_pivot_loc = i
			end
		end
			old = P[k,:]
			P[k,:] .= P[biggest_pivot_loc,:]
			P[biggest_pivot_loc,:] .= old
			B .= P * A
	end
		
	(L, U) = computeLU(P * A)	
	return (L, U, P)
end
"""
	LUPsolve(A, b) 

Computes the LUP factorization of A, then solves the matrix equation Ax = b
for x and returns x

# Examples
julia> A = [2 0 0; 0 2 0; 0 0 2]
julia> b = [2, 2, 2]
julia> x = LUPsolve(A, b)
julia> @assert x = [1, 1, 1]
"""
function LUPsolve(A, b, L, U, P)
	N = size(A)[1]
	C = Array{Float64}(undef, N)
	permuted = P * b

	"""
	This loop solves the matrix equation Lc = Pb through back substitution.
	"""
	for k = 1:N 
		c_entry = permuted[k]
		for i=1:k-1
			c_entry -= C[i] * L[k, i]
		end
		C[k] = c_entry
	end

	X = Array{Float64}(undef, N)

	"""
	This loop solves the matrix equation Ux = C through back substitution.
	"""
	for k in N:-1:1 
		x_entry = C[k]
		for i in N:-1:k+1
			x_entry -= X[i] * U[k, i]
		end
		x_entry /= U[k, k]
		X[k] = x_entry
	end

	return X
end

function conj_grad(A, guess, b, tolerance, max_itters)
	r = -b
	ρ = 0
	p = 0
	for i in 1:max_itters
		last_ρ	= ρ	
		z = r
		ρ = dot(z, r)
		if i == 1
			p = z
		else
			B = ρ/last_ρ
			p = z + (B * p)
		end
		q = A * p
		δ = ρ/dot(p, q)
		guess = guess - (δ * p)
		r = r - (δ * q)
		if tolerance >= (norm((A * guess) - b)/norm(guess))
			return guess
		end
	end
	return 0 
end

function comp_piv_matrices(p, q, r)
	N = length(p)

	Id = sparse(1:N, 1:N, ones(N), N, N)
	P = copy(Id)
	Q = copy(Id)

	for i = 1:N
		P[i,:] = Id[p[i],:]
		Q[:,i] = Id[:,q[i]]
	end

	R = sparse(1:N,1:N,r,N,N)

	return (P,Q,R)
end

for N in [10, 100, 1000]
	A = Array{Float64}(undef, N, N)
	b = Array{Float64}(undef, N, 1)
	A .= rand(N,N)
	b .= rand(N, 1)

	println("Compute my LUP factors")
	@time (myL, myU, myP) = computeLUP(A)
	@assert myL * myU ≈ myP * A

	println("Compute my solution")
	@time x = LUPsolve(A, b, myL, myU, myP)
	@assert A * x ≈	b

	positive_definite = transpose(A) * A
	guess = zeros(N)
	tolerance = 0.00000001
	max_itters = 3 * N
	@time guess = conj_grad(positive_definite, guess, b, tolerance, max_itters)
	residual = (positive_definite * guess) - b
	positive_residual = broadcast(abs, residual)
	@assert sum(positive_residual) < 0.01
end

N = 1000
A = zeros(N, N)
A[1,1] = -2
A[1,2] = 1
for i in 2:N-1
	A[i, i] = -2
	A[i,i+1] = 1
	A[i,i-1] = 1
end
A[N, N] = -2
A[N, N-1] = 1
b = Array{Float64}(undef, N, 1)
b .= rand(N, 1)

println("Compute my LUP factors, dense")
@time F = lu(A)
myL = F.L
myU = F.U
myP = F.P
println("Compute my solution, dense")
@time c = myL \ (myP * b)
@time x = myU \ c
@assert A * x ≈	b

rows = Float64[]
cols = Float64[]
vals = Float64[]
push!(rows, 1)
push!(rows, 1)
push!(cols, 1)
push!(cols, 2)
push!(vals, -2)
push!(vals, 1)

for i in 2:N-1
	push!(rows, i)
	push!(rows, i)
	push!(rows, i)
	push!(cols, i-1)
	push!(cols, i)
	push!(cols, i+1)
	push!(vals, 1)
	push!(vals, -2)
	push!(vals, 1)
end
push!(rows, N)
push!(rows, N)
push!(cols, N-1)
push!(cols, N)
push!(vals, 1)
push!(vals, 1)
sparseB = sparse(b)
sparseA = sparse(rows, cols, vals)
println("Compute my LUP factors, sparse")
@time F = lu(sparseA)
myL = F.L
myU = F.U
(P,Q,R) = comp_piv_matrices(F.p, F.q, F.Rs)

println("Compute my solution, sparse")
@time x = Q * (F.U\(F.L\(P*R*sparseB)))
@assert sparseA * x ≈  sparseB
