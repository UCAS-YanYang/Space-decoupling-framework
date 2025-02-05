function problem = completion_Mh(r, omega, data, manifoldH)
% Low-rank completion with additional constraint X\in\H:
%
%     min   f(X) = 1/2 \|mask.*(X - A)\|_F^2
%     s.t.  rank(X)\le r,  X\in H
%
% Create the Manopt problem on the space-decoupling geometry M_h.
%
% r is the rank parameter
% omega is the metric parameter.
% data is an masked matrix, with the original one in Ob(m,n).
% manifoldH represents the manifold H, the level set of an orthogonally
% invariant function, which provides the following two tools:
%   1.proj(h,u): project u onto the tangent space to H at point h
%   2.projH(y): project y onto H

    if ~exist('omega', 'var') || isempty(omega)
        omega = 0.5;
    end

    m = data.m;
    n = data.n;
    A = data.A;
    I = data.I;
    J = data.J;
    mask = data.mask;
    
    A_test = data.A_test;
    I_test = data.I_test;
    J_test = data.J_test;
    mask_test = data.mask_test;

    M = Mhfactory(m, n, r, omega, manifoldH);
    problem.M = M;

    problem.name = sprintf('spherical data fitting problem m=%d n=%d r=%d omega=%d', m, n, r, omega);

    function store = prepareinvSf(X, store)
        if ~isfield(store, 'invSf')
            store.invSf = M.sfactorinv(X);
        end
    end

    function store = prepareUS(X, store)
        if ~isfield(store, 'US')
            store.US = X.U * X.S;
        end
    end

    function store = prepareegrad(X, store)
        if ~isfield(store, 'egrad')
            store = prepareUS(X, store);
            store.egrad = spmaskmult(store.US, X.V', I, J) - A;
        end
    end

    function store = prepareUSinvSf(X, store)
        if ~isfield(store, 'USinvSf')
            store = prepareUS(X, store);
            store = prepareinvSf(X, store);

            store.USinvSf = store.US * store.invSf;
        end
    end

    problem.cost = @cost;
    function [f, store] = cost(X, store)
        store = prepareegrad(X, store);

        f = .5*norm(store.egrad, 'fro')^2;

        store = incrementcounter(store, 'costcalls');
    end

    problem.cost_test = @cost_test;
    function [f, store] = cost_test(X, store)
        store = prepareUS(X, store);
        test_grad = spmaskmult(store.US, X.V', I_test, J_test) - A_test;
        f = norm(test_grad, 'fro')/norm(A_test, 'fro');
    end

    problem.cost_train = @cost_train;
    function [f, store] = cost_train(X, store)
        store = prepareegrad(X, store);
        train_grad = store.egrad;
        f = norm(train_grad, 'fro')/norm(A, 'fro');
    end

    problem.grad = @grad;
    function [G, store] = grad(X, store)
        store = prepareegrad(X, store);
        store = prepareinvSf(X, store);
        store = prepareUS(X, store);
        store = prepareUSinvSf(X, store);

        ZUSinvSf = multfullsparse(store.USinvSf', store.egrad, mask)';

        G.K = multsparsefull(store.egrad, X.V, mask);
        G.K = manifoldH.proj(store.US,G.K);         % #H
        G.Vp = ZUSinvSf - X.V * (X.V' * ZUSinvSf);

        store = incrementcounter(store, 'gradcalls');
    end

    problem.hess = @hess;
    function [H, store] = hess(X, Xd, store)
        store = prepareegrad(X, store);
        store = prepareinvSf(X, store);
        store = prepareUS(X, store);
        store = prepareUSinvSf(X, store);

        
        spXd = spmaskmult(Xd.K, X.V', I, J) ...
            + spmaskmult(store.US, Xd.Vp', I, J);
        
        %%%%%%%% #H
        XdV = multsparsefull(spXd, X.V, mask);
        XdV_1 = manifoldH.proj(store.US,XdV);
        ZV = multsparsefull(store.egrad, X.V, mask);
        XdV_2 = (sum(store.US.*ZV,2)).*Xd.K;
        XdV =  XdV_1 -  XdV_2;
        %%%%%%%% #H
        
        ZVp = multsparsefull(store.egrad, Xd.Vp, mask);

        H.K = XdV + ZVp - store.USinvSf * (store.US' * ZVp);
        H.K = manifoldH.proj(store.US,H.K);

        MK = Xd.K - store.USinvSf * (store.US' * Xd.K);
        ZtMK = multfullsparse(MK', store.egrad, mask)';
        XdtUS = multfullsparse(store.US', spXd, mask)';
        
        % modification
        nablaK = multsparsefull(store.egrad, X.V, mask);
        modification = -Xd.Vp*((nablaK- manifoldH.proj(store.US,nablaK))'*store.US);
        
        
        W = (XdtUS + ZtMK + modification) * store.invSf;

        H.Vp = W - X.V * (X.V' * W);

        store = incrementcounter(store, 'hesscalls');
    end

end
