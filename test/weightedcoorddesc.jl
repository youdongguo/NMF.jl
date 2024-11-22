function weightedobjective(X, M, W_v, H_v, r)
    m, n = size(X)
    W = reshape(W_v, m, r)
    H = reshape(H_v, r, n)
    return 0.5*sum(M.^2 .* (X-W*H).^2)
end

atminimum(x, slope; xtol=eps(typeof(x)), slopetol=eps(typeof(slope))) = x<=xtol ? slope >= -slopetol : abs(slope) <= slopetol

@testset "weightedcoorddesc" begin
    for T in (Float64, Float32)
        X, Wg, Hg = laurberg6x3(T(0.3))
        M = ones(T, size(X)...)
        W = Wg .+ rand(T, size(Wg)...)*T(0.1)
        NMF.solve!(NMF.WeightedCoordinateDescent{T}(M, maxiter=10^8, tol=1e-9), X, W, Hg)
        @test X ≈ W * Hg atol=1e-4

        weightedobjective_w(w) = weightedobjective(X, M, w, Hg, size(Hg, 1))
        weightedobjective_h(h) = weightedobjective(X, M, W, h, size(W, 2))
        grad_w = ForwardDiff.gradient(weightedobjective_w, W[:])
        grad_h = ForwardDiff.gradient(weightedobjective_h, Hg[:])
        
        for (wi, gi) in zip(W[:], grad_w)
            @test atminimum(wi, gi; xtol=2e-7, slopetol=2e-7)
        end

        for (hi, gi) in zip(Hg[:], grad_h)
            @test atminimum(hi, gi; xtol=2e-7, slopetol=2e-7)
        end
    end
end

@testset "gradient in update rule" begin
    function weightedresidual(X, M, W, H, j, w_j, h_j)
        R_j = X-W*H+W[:,j]*H[j,:]'
        E = 0
        for p in axes(X, 1), q in axes(X, 2)
            E += M[p,q]^2*(R_j[p,q]-w_j[p]*h_j[q])^2
        end
        return E
    end

    X = rand(12,10);
    M = rand(eltype(X), size(X));
    W = rand(12,8);
    H = rand(8,10);
    for j in 1:8
        global w_j = W[:,j]
        global h_j = H[j,:]
        weightedresidual_w(w) = weightedresidual(X, M, W, H, j, w, h_j)
        w1 = rand(length(w_j))
        global grad_w = .5 .*ForwardDiff.gradient(weightedresidual_w, w1)
        global grad_w1 = ((M.^2)*(h_j.^2)).*w1 .- (M.^2 .*(X-W*H+w_j*h_j'))*h_j
        @test sum(abs2, grad_w.-grad_w) < 1e-12
    end

    for j in 1:8
        global w_j = W[:,j]
        global h_j = H[j,:]
        weightedresidual_h(h) = weightedresidual(X, M, W, H, j, w_j, h)
        h1 = rand(length(h_j))
        global grad_h = .5 .*ForwardDiff.gradient(weightedresidual_h, h1)
        global grad_h1 = ((M.^2)'*(w_j.^2)).*h1 .- (M.^2 .*(X-W*H+w_j*h_j'))'*w_j
        @test sum(abs2, grad_h.-grad_h1) < 1e-12
    end
end
