function M = Mhfactory(m, n, r, omega, manifoldH)
% Manifold to optimize over bounded-rank matrices with additional constraints
% through the space-decoupling parameterization (M_h,\phi)
%
% function M = Mhfactory(m, n, r, omega, manifoldH)
% manifoldH represents the manifold H, the level set of an orthogonally
% invariant function, which provides the following two tools:
%   1.proj(h,u): project u onto the tangent space to H at point h
%   2.projH(y): project y onto H
%
% The embedding space is E = R^(mxn) x Sym(n) where Sym denotes the set of
% symmetric matrices.
% Let Gr(n, s) be the Grassmannian: orthogonal projectors of size n and of
% rank s. The manifold, associated with space-decoupling parameterization, is 
%
% M_h = {(X, G) in E such that G is in Gr(n, n - r) and X*G = 0 and h(X)=0}
%
% The condition X*G = 0 implies that X has rank at most r.
%
% A point (X, G) in M_h is represented as a structure with three fields:
%
%     H, V  such that  X = H*V'  and  P = I - V*V'.
%
% A tangent vector at (X, G) is represented as a structure with two fields:
%
%     K, Vp  such that  Xdot = K*V' + H*Vp'  and  Gdot = -Vp*V' - V*Vp'.
%
% The matrix K (mxr) in tangent space to H^r; while Vp (nxr) satisfies V'*Vp = 0.
%
% We equip the embedding space E with the metric
%
%     inner((Xd1, Pd1), (Xd2, Pd2)) = Tr(Xd1'*Xd2) + omega * Tr(Pd1'*Pd2)
%
% for some parameter omega > 0 (the default is 1/2).
%
% The tangent spaces of M_h inherit this metric, so that M is a Riemannian
% submanifold of E.
%
% Two retractions are available:
% - retr_first (first-order).
% - retr_polar (second-order).
% The defaults are M.retr = @retr_polar.
%

    if ~exist('omega', 'var') || isempty(omega)
        omega = .5;
    end

    assert(r <= min(m, n), 'The rank r should be <= min(m, n).');
    assert(omega >= 0, 'omega should be positive (default is 1/2).');
    if omega == 0
        warning('Mh:omegazero', ...
                ['omega was set to 0. It should be positive.\n' ...
                'Disable this warning with ' ...
                'warning(''off'', ''Mh:omegazero'').']);
    end

    % The embedding space consists of potentially large matrices.
    % We use euclideanlargefactory to allow efficient representations.
    Emn = euclideanlargefactory(m, n);
    Enn = euclideanlargefactory(n, n);

    M.manifoldH = manifoldH;
    
    M.omega = omega;

    M.name = @() sprintf(['Space-decoupling manifold Mh of '
                          '%dx%d matrices with rank bounded '
                          'by %d with omega = %g'], m, n, r, omega);

    M.dim = @() (m + n - r)*r;

    sfactor = @(XP) 2*omega*eye(r) + XP.S^2;
    sfactorinv = @(XP) diag(1./diag(sfactor(XP)));
    M.sfactor = sfactor;
    M.sfactorinv = sfactorinv;

    % Usual trace inner product of two matrices.
    matinner = @(A, B) A(:)'*B(:);

    M.inner = @(XP, XPdot1, XPdot2) matinner(XPdot1.K, XPdot2.K) + ...
                                matinner(XPdot1.Vp, XPdot2.Vp*sfactor(XP));

    M.norm = @(XP, XPdot) sqrt(max(0, M.inner(XP, XPdot, XPdot)));

    M.typicaldist = @() M.dim();

    % Given XPdot in tangent vector format, projects the component K and Vp 
    % such that they satisfy the tangent space constraints up to numerical
    % errors.
    M.tangent = @tangent;
    function XPdot = tangent(XP, XPdot)
        XPdot.Vp = XPdot.Vp - XP.V*(XP.V'*XPdot.Vp);
        XPdot.K = M.manifoldH.proj(XP.U*XP.S,XPdot.K);
    end

    % XPa is in the embedding space E, that is, it is a struct with fields:
    %   XPa.X  --  an mxn matrix in euclideanlargefactory format, Emn
    %   XPa.P  --  an nxn matrix in euclideanlargefactory format, Enn
    % This function projects XPa to the tangent space at XP.
    % The output is in the tangent vector format.
    M.proj = @projection;
    function XPdot = projection(XP, XPa)
        % In principle, XPa.P should already be symmetric.
        % We take the symmetric part (times 2) to be permissive, but if
        % this becomes a performance issue there is something to gain here.
        % In matrix format, symPV equals (XPa.P + XPa.P.')*XP.V;
        symPV = Enn.times(XPa.P, XP.V) + Enn.transpose_times(XPa.P, XP.V);

        % In matrix format, the first term is XPa.X.'*XP.U*XP.S;
        B = Emn.transpose_times(XPa.X, XP.U*XP.S) - omega*symPV;
        B = B / sfactor(XP);

        XPdot.K = Emn.times(XPa.X, XP.V); % = XPa.X*XP.V in matrix format
        XPdot.K = M.manifoldH.proj(XP.U*XP.S,XPdot.K);
        XPdot.Vp = B - XP.V*(XP.V.'*B);
    end

    % First-order retraction based on retractions on .
    M.retr_first = @retr_first;
    function XPnew = retr_first(XP, XPdot, t)
        if nargin < 3
            t = 1;
        end
%         
        Z = XP.V + t*XPdot.Vp;
        Vnew = Z/sqrtm(Z'*Z);                   % polar
        US = XP.U*XP.S;
        H_new = M.manifoldH.projH(US+t*XPdot.K);
        % H_new has size m-by-r
        [HU, HS, HV] = svd(H_new, 'econ');
        XPnew.U = HU;
        XPnew.S = HS;
        XPnew.V = Vnew * HV;
    end

    % Second-order retraction based on the polar retraction.
    M.retr_polar = @retr_polar;
    function XPnew = retr_polar(XP, XPdot, t)
        if nargin < 3
            t = 1;
        end
        
        US = XP.U*XP.S;
        
        KtUS = (XPdot.K'*US) / sfactor(XP);
        Z = XP.V + t*XPdot.Vp*(eye(r) - t*KtUS);
        Vnew = Z/sqrtm(Z'*Z);
      
        USptK = US + t*XPdot.K;
        temp1 = USptK*(XP.V'*Vnew);
        temp2 = t*US*(XPdot.Vp'*Vnew);
        H_new = M.manifoldH.projH(temp1+temp2);
        
        % H_new has size m-by-r
        [HU, HS, HV] = svd(H_new, 'econ');
        XPnew.U = HU;
        XPnew.S = HS;
        XPnew.V = Vnew * HV;
    end

    % Multiple retractions are available for the desingularization.
    % We choose default first- and second-order retractions here.
    M.retr = M.retr_first;
    M.retr = M.retr_polar;

    % Same hash as fixedrankembeddedfactory.
    M.hash = fixedrankembeddedfactory(m, n, r).hash;

    % Generate a random point on M.
    % The factors U and V are sampled uniformly at random on Stiefel.
    % The singular values are uniform in [0, 1].
    M.rand = @random;
    function XP = random()
        XP.U = qr_unique(randn(m, r));
        XP.V = qr_unique(randn(n, r));
        XP.S = diag(sort(rand(r, 1), 'descend'));
        
        normalized_H = manifoldH.projH(XP.U* XP.S);
        [XP.U,XP.S,V] = svd(normalized_H,'econ');
        XP.V = XP.V*V;
    end

    % Generate a unit-norm random tangent vector at XP.
    % Note: this may not be the uniform distribution.
    M.randvec = @randomvec;
    function XPdot = randomvec(XP)
        XPdot.K  = randn(m, r);
        XPdot.K = manifoldH.proj(XP.U* XP.S,XPdot.K);
        XPdot.Vp = randn(n, r);
        XPdot = tangent(XP, XPdot);
        normXPdot = M.norm(XP, XPdot);
        XPdot.K = XPdot.K/normXPdot;
        XPdot.Vp = XPdot.Vp/normXPdot;
    end

    % Linear combination of tangent vectors.
    % Returns the tangent vector a1*XPdot1 + a2*XPdot2.
    M.lincomb = @lincomb;
    function XPdot3 = lincomb(~, a1, XPdot1, a2, XPdot2)
        if nargin == 3
            XPdot3.K = a1*XPdot1.K;
            XPdot3.Vp = a1*XPdot1.Vp;
        elseif nargin == 5
            XPdot3.K  = a1*XPdot1.K + a2*XPdot2.K;
            XPdot3.Vp = a1*XPdot1.Vp + a2*XPdot2.Vp;
        else
            error('Mhfactory.lincomb takes 3 or 5 inputs.');
        end
    end

    M.zerovec = @(XP) struct('K', zeros(m, r), 'Vp', zeros(n, r));

    % The function 'vec' is isometric from the tangent space at XP to real
    % vectors of length (m+n+r)r.
    M.vec = @vec;
    function XPdotvec = vec(XP, XPdot)
        VpS = XPdot.Vp*sqrt(sfactor(XP));
        XPdotvec = [XPdot.K(:); VpS(:)];
    end

    % The function 'mat' is the left-inverse of 'vec'. It is sometimes
    % useful to apply 'tangent' to the output of 'mat'.
    M.mat = @mat;
    function XPdot = mat(XP, v)
        K = reshape(v(1:(m*r)),  [m, r]);
        VpS = reshape(v((m*r)+(1:(n*r))), [n, r]);
        XPdot = struct('K', K, 'Vp', VpS/sqrt(sfactor(XP)));
    end

    M.vecmatareisometries = @() true;

    % It is sometimes useful to switch between representation of matrices
    % as triplets or as full matrices of size m x n. The function to
    % convert a matrix to a triplet, matrix2triplet, allows to specify the
    % rank of the representation. By default, it is equal to r. Omit the
    % second input (or set to inf) to get a full SVD triplet (in economy
    % format). If so, the resulting triplet does not represent a point on
    % the manifold.
    M.matrix2triplet = fixedrankembeddedfactory(m, n, r).matrix2triplet;
    M.triplet2matrix = fixedrankembeddedfactory(m, n, r).triplet2matrix;


    % Transform a tangent vector XPdot represented as a structure (K, Vp)
    % into a structure that represents that same tangent vector in the
    % ambient space E.
    % Returns a struct XPa with two fields:
    %     XPa.X represents XPdot.K*XP.V' + XP.U*XP.S*XPdot.Vp'  of size mxn
    %     XPa.P represents -XPdot.Vp*XP.V' - XP.V*XPdot.Vp'     of size nxn
    % The representations follow euclideanlargefactory formats Emn and Enn.
    M.tangent2ambient_is_identity = false;
    M.tangent2ambient = @tangent2ambient;
    function XPa = tangent2ambient(XP, XPdot)
        XPa.X = struct('L', [XPdot.K, XP.U*XP.S], ...
                       'R', [XP.V, XPdot.Vp]);
        XPa.P = struct('L', -[XPdot.Vp, XP.V], ...
                       'R',  [XP.V, XPdot.Vp]);
    end

    % Vector transport extending the space-decoupling spirit.
    M.transp = @transporter;
    function XP2dot = transporter(XP1, XP2, XP1dot)
        XP2dot.K = manifoldH.proj(XP2.U* XP2.S,XP1dot.K);
        XP2dot.Vp = XP1dot.Vp - XP2.V*(XP2.V'*XP1dot.Vp);
    end

end
