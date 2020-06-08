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

function knl_gemv(A, x, b, y)
    (R, C) = size(A)
    @assert R = size(y)
    @assert C = size(x)
    bidx = blockIdx().x
    tidx = threadIdx().x
    dimx = blockDim().x

    bidy = blockIdx().y
    tidy = threadIdx().y
    dimy = blockDim().y
    i = dimx * (bidx - 1) + tidx
    j = dimy * (bidy - 1) + tidy
    if i <= size(x) && j <= size(y)
        y[i] += x[j] * A[i,j] + (b[j] / C)
    return nothing
end

R = 5
C = 5
A = [1.0 0.0 0.0 0.0 0.0 ; 0.0 1.0 0.0 0.0 0.0; 0.0 0.0 1.0 0.0 0.0 ; 0.0 0.0 0.0 1.0 0.0 ; 0.0 0.0 0.0 0.0 1.0]
for i = 1:R
    for j = 1:C
        if i == j
            A[i,j] = 1
        else
            A[i,j] = 0
        end
    end
end

b = Array{Float64}(undef, 5)
x = Array{Float64}(undef, 5)
y = Array{Float64}(undef, 5)

for i = 1:5
    b[i] = 1
    x[i] = 1
    y[i] = 0
end


D_A = CuArray(A)
d_x = CuArray(x)
d_v = CuArray(v)
d_y = CuArray(y)

threads_per_x = 32
threads_per_y = 32
thd_tuple = (threads_per_x, threads_per_y)
num_blocks_x = cld(R, threads_per_x)
num_blocks_y = cld(C, threads_per_y)

truth = [2.0 2.0 2.0 2.0 2.0]

@cuda threads = thd_tuple blocks = (num_blocks_x, num_blocks_y) knl_gemv!(D_A, d_x, b_x, y_x)

synchronize()

@show y
@assert y ≈ truth
