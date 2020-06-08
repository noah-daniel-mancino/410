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

    i = dimx * (bidx - 1) + tidx
    if i <= R
        for j = 1:C
        y[i] += (x[j] * A[i,j]) + (b[j] / C)
        end
	end
    return nothing
end

R = 25000
C = 25000
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
compile_trash = Array{Float64}(undef, C)

for i = 1:C
    b[i] = 1
    x[i] = 1
    y[i] = 0
    compile_trash[i] = 0
end


D_A = CuArray(A)
d_x = CuArray(x)
d_b = CuArray(b)
d_y = CuArray(y)
trash = CuArray(compile_trash)

threads_per_x = 32
num_blocks_x = cld(R, threads_per_x)
@cuda threads=threads_per_x blocks=num_blocks_x knl_gemv!(D_A, d_x, d_b, trash)
t_dev = @elapsed begin
@cuda threads=threads_per_x blocks=num_blocks_x knl_gemv!(D_A, d_x, d_b, d_y) 
synchronize()
end
@show t_dev
@time fake_knl_gemv!(A, x, b, y)
