mutable struct WeightedCoordinateDescent{T}
    M::Matrix{T}
    maxiter::Int           # maximum number of iterations (in main procedure)
    verbose::Bool          # whether to show procedural information
    tol::T                 # tolerance of changes on W and H triggering convergence
    update_H::Bool         # whether to update H
    shuffle::Bool          # # if true, randomize the order of coordinates in the CD solver           # mask matrix
    function WeightedCoordinateDescent{T}(M::AbstractMatrix;
                                        maxiter::Integer=100,
                                        verbose::Bool=false,
                                        tol::Real=cbrt(eps(T)),
                                        update_H::Bool=true,
                                        shuffle::Bool=false) where T
        new{T}(M, maxiter, verbose, tol, update_H, shuffle)
    end
end


solve!(alg::WeightedCoordinateDescent{T}, X, W, H) where {T} =
    nmf_skeleton!(WeightedCoordinateDescentUpd{T}(alg.M, alg.shuffle, alg.update_H),
                  X, W, H, alg.maxiter, alg.verbose, alg.tol)


struct WeightedCoordinateDescentUpd{T} <: NMFUpdater{T}
    M::Matrix{T}
    shuffle::Bool
    update_H::Bool
    function WeightedCoordinateDescentUpd{T}(M::Matrix{T}, shuffle::Bool, update_H::Bool) where {T}
        new{T}(M,
               shuffle,
               update_H)
    end
end

mutable struct WeightedCoordinateDescentState{T}
    WH::Matrix{T}

    function WeightedCoordinateDescentState{T}(W, H) where T
        new{T}(W*H)
    end
end

prepare_state(::WeightedCoordinateDescentUpd{T}, ::AbstractArray{T}, W, H) where T = WeightedCoordinateDescentState{T}(W, H)

function evaluate_objv(u::WeightedCoordinateDescentUpd{T}, s::WeightedCoordinateDescentState{T}, X, W, H) where T
    M = u.M
    mul!(s.WH, W, H)
    sqL2dist(M.*X, M.*s.WH) / 2
end

"Updates W only"
function _update_weighted_coord_descent!(X::AbstractArray{T}, M::AbstractArray{T}, W::AbstractArray{T}, H::AbstractArray{T}) where T
    ncomponents = size(W, 2)
    m, n = size(M)
    numerator = zeros(T, m)
    denominator = zeros(T, m)
    for t in 1:ncomponents
        fill!(numerator, zero(T))
        fill!(denominator, zero(T))
        for i in 1:m
            for k in 1:n
                M_ik2 = M[i,k]^2
                numerator[i] += M_ik2*X[i, k]*H[t,k]
                for r in 1:ncomponents
                    r == t && continue
                    numerator[i] -= M_ik2*W[i,r]*H[r,k]*H[t,k]
                end
                denominator[i] += M_ik2*H[t,k]^2
            end
            W[i,t] = max(numerator[i] / denominator[i], zero(eltype(W)))
        end
    end
    return W
end

function update_wh!(upd::WeightedCoordinateDescentUpd{T}, ::WeightedCoordinateDescentState{T}, X::AbstractArray{T}, W::AbstractArray{T}, H::AbstractArray{T}) where T
    _update_weighted_coord_descent!(X, upd.M, W, H)

    # update H
    if upd.update_H
        Wt = transpose(W)
        Ht = transpose(H)
        Xt = transpose(X)
        Mt = transpose(upd.M)
        _update_weighted_coord_descent!(Xt, Mt, Ht, Wt)
    end
end