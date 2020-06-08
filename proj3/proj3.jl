using CuArrays
using CUDAnative
using CUDAdrv
using LinearAlgebra

function fake_knl_gemv!(A, x, b, y)
    (R, C) = size(A)
    for i = 1:R
        for j = 1:C
            y[i] += (x[j] * A[i,j]) + (b[j] / C)
        end
    end
    return nothing
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

R = 5
C = 5
A = Array{Float64}(undef, R, C)
for i = 1:R
    for j = 1:C
        if i == j
            A[i,j] = 2
        else
            A[i,j] = 0
        end
    end
end

b = Array{Float64}(undef, C)
x = Array{Float64}(undef, C)
y = Array{Float64}(undef, C)

for i = 1:C
    b[i] = 1
    x[i] = 1
    y[i] = 0
end


D_A = CuArray(A)
d_x = CuArray(x)
d_b = CuArray(b)
d_y = CuArray(y)

threads_per_x = 32
num_blocks_x = cld(R, threads_per_x)

@time @cuda threads=thd_tuple blocks=num_blocks_x knl_gemv!(D_A, d_x, d_b, d_y) 
synchronize()
@time fake_knl_gemv!(A, x, b, y)
