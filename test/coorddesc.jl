# some tests for CoordinateDescent
import Base: *, Array, size, length, getindex, similar, adjoint, transpose
import LinearAlgebra: mul!, dot

@testset "coorddesc" begin
    for T in (Float64, Float32)
        X, Wg, Hg = laurberg6x3(T(0.3))
        W = Wg .+ rand(T, size(Wg)...)*T(0.1)
        NMF.solve!(NMF.CoordinateDescent{T}(α=0.0, maxiter=1000, tol=1e-9), X, W, Hg)
        @test X ≈ W * Hg atol=1e-4

        # Regularization
        X, Wg, Hg = laurberg6x3(T(0.3))
        W = Wg .+ rand(T, size(Wg)...)*T(0.1)
        NMF.solve!(NMF.CoordinateDescent{T}(α=1e-4, l₁ratio=0.5, shuffle=true, maxiter=1000, tol=1e-9), X, W, Hg)
        @test X ≈ W * Hg atol=1e-2
    end
end

# struct FactoredMatrix{T} <: Factorization{T}
#     U::Matrix{T}      # M x K
#     V::Matrix{T}      # K x N
#     temp::Matrix{T}   # K x J
#     tempT::Matrix{T}  # J x K

#     function FactoredMatrix{T}(U::Matrix{T}, V::Matrix{T}, J::Int) where T
#         K = size(U, 2)
#         size(V, 1) == K || throw(DimensionMismatch("U and V must be able to be multiplied"))
#         temp  = similar(U, K, J)
#         tempT = similar(U, J, K)
#         new{T}(U, V, temp, tempT)
#     end
# end
# FactoredMatrix(U::Matrix{T}, V::Matrix{T}, J::Integer) where {T} = FactoredMatrix{T}(U, V, Int(J))

# size(A::FactoredMatrix) = size(A.U,1), size(A.V,2)
# size(A::FactoredMatrix, d::Integer) = d == 1 ? size(A.U,1) : (d == 2 ? size(A.V,2) : 1)
# similar(A::FactoredMatrix, dims) = Array{eltype(A.U)}(undef, dims)
# similar(A::FactoredMatrix, T, dims) = Array{T}(undef, dims)
# length(A::FactoredMatrix) = size(A.U,1)*size(A.V,2)

# Array(A::FactoredMatrix) = A.U*A.V

# adjoint(A::FactoredMatrix) = Adjoint(A)
# getindex(A::Adjoint{T,FactoredMatrix{T}},i,j) where T =
#     um(A.parent.U[j,:].*A.parent.V[:,i])'
# transpose(A::FactoredMatrix) = Transpose(A)
# getindex(A::Transpose{T,FactoredMatrix{T}},i,j) where T =
#     sum(A.parent.U[j,:] .* A.parent.V[:,i])

# mul!(C, A::FactoredMatrix, B::FactoredMatrix) = error("not defined")
# LinearAlgebra.mul!(C, A::FactoredMatrix, B) = mul!(C, A.U, mul!(A.temp, A.V, B))

# *(A::FactoredMatrix, B::FactoredMatrix) = error("not defined")
# *(A::FactoredMatrix, B) =
#     mul!(Array{promote_type(eltype(A),eltype(B))}(undef, (size(A,1), size(B,2))), A, B)
# *(A, B::FactoredMatrix) =
#     mul!(Array{promote_type(eltype(A),eltype(B))}(undef, (size(A,1), size(B,2))), A, B)

# function LinearAlgebra.issymmetric(A::FactoredMatrix)
#     if size(A.U,1) != size(A.V,2) || size(A.U,2) != size(A.V,1)
#         return false
#     end
#     A.U == A.V'
# end

# # For computing sum-of-squared difference between two FactoredMatrixes
# function dot(A::FactoredMatrix, B::FactoredMatrix)
#     M1 = B.U'*A.U
#     M2 = B.V*A.V'
#     sum(M1.*conj(M2))
# end

# ssd(A::FactoredMatrix, B::FactoredMatrix) = dot(A, A) - 2*real(dot(A, B)) + dot(B, B)
# StatsBase.sqL2dist(A::FactoredMatrix, B) = sqL2dist(A.U*A.V, B)

# @testset "coorddesc_factored_matrix" begin
#     for T in (Float64, Float32)
#         WgHg, Wg, Hg = laurberg6x3(T(0.3))
#         X = FactoredMatrix(Wg, Hg, size(Wg,2))
#         W = Wg .+ rand(T, size(Wg)...)*T(0.1)
#         NMF.solve!(NMF.CoordinateDescent{T}(α=0.0, maxiter=1000, tol=1e-9), X, W, Hg)
#         @test WgHg ≈ W * Hg atol=1e-4

#         # Regularization
#         WgHg, Wg, Hg = laurberg6x3(T(0.3))
#         X = FactoredMatrix(Wg, Hg, size(Wg,2))
#         W = Wg .+ rand(T, size(Wg)...)*T(0.1)
#         NMF.solve!(NMF.CoordinateDescent{T}(α=1e-4, l₁ratio=0.5, shuffle=true, maxiter=1000, tol=1e-9), X, W, Hg)
#         @test WgHg ≈ W * Hg atol=1e-2
#     end
# end


