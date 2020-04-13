using LinearAlgebra

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
	ell = copy(Id) # the L in the LU decomposition
	ell_inv = copy(Id)
	Atilde = copy(A)
	L = copy(Id)

	for k = 1:N-1 # marching through columns
		
		ell .= Id
		ell_inv .= Id
		for i = k+1:N # marching through rows under given pivot
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
	for k = 1:N-1 # march through columns.
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
function LUPsolve(A, b)
	N = size(A)[1]
	(L, U, P) = computeLUP(A)
	C = Array{Float64}(undef, N)
	permuted = P * b

	"""
	This loop solves the matrix equation Lc = Pb through back substitution.
	"""
	for k = 1:N # marching through rows
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
	for k in N:-1:1 # marching through rows
		x_entry = C[k]
		for i in N:-1:k+1
			x_entry -= X[i] * U[k, i]
		end
		#= In the Lc case, diagnol are always 1, so we only did this implicity =#
		x_entry /= U[k, k]
		X[k] = x_entry
	end

	return X
end


N = 10
A = Array{Float64}(undef, N, N)
A .= rand(N,N)

println("Compute my LU fact")
@time (myL, myU, myP) = computeLUP(A)
@assert myL * myU ≈ myP * A

D = [0 1 0; 1 0 0; 0 0 1]
d = [1, 2, 3]
x = LUPsolve(D, d)
"""
b = rand(N,1) # defines the right hand side of Ax = b

x = LUPsolve(myL, myU, b)
"""
