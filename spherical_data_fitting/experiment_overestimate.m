function experiment_overestimate(ratio,r)
    clear;
    clc;
    
    task = 'overestimate/';
    
    
    m = 5000;
    n = 6000;
    rstar = 6;
    r = 10;
    
    ratio = 0.1;
    omega_list = [0.1,0.5, 10,50];
    
    
    options.verbosity = 2;
    options.maxiter = 500;
    options.maxtime = 200;
    options.tolgradnorm = 1e-13;
    options.theta = sqrt(2) - 1.2; % theta parameter for tCG
    
    % record test error
    options.statsfun = @statsfun_test_error;


    %%%%%%%%%%%%%%%% Define the manifold H %%%%%%%%%%%%%%%%
    %   1.proj(h,u): project u onto Th(H)
    %   2.projH(y): project y onto H
    %   3.hess(h,u,egradf,ehessf[u]): return Riemannian hessian-vector
    manifoldH.proj = @manifoldproj;
    manifoldH.projH = @manifoldprojH;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    fprintf('Running experiment with ratio = %.4f\n', ratio);
    rho = ratio * m * n;


    data.m = m;
    data.n = n;
    [data.I, data.J, data.mask,~,data.I_test, data.J_test, data.mask_test,~] = randmask(m, n, rho,1);
    AU = stiefelfactory(m, rstar).rand();
    AS = diag(rand(rstar, 1));
    normalized_H = manifoldH.projH(AU * AS);
    AV = stiefelfactory(n, rstar).rand();
    data.A = spmaskmult(normalized_H, AV', data.I, data.J);
    data.A_test = spmaskmult(normalized_H, AV', data.I_test, data.J_test);

    problems = cell(1, numel(omega_list));
    for i = 1:numel(omega_list)
        problems{i} = completion_Mh(r, omega_list(i), data, manifoldH);
    end


    T = normalized_H*AV';
    randomIndices = randperm(n, r);
    H_padded =  manifoldH.projH(T(:,randomIndices));
%     H_padded= manifoldH.projH(rand(m,r));
    [U1,S1,~] = svd(H_padded,'econ');   
    AV = stiefelfactory(n, r).rand();
    
    X0.U = U1;
    X0.S = S1;
    X0.V = AV;
    X0s = {X0,X0,X0,X0};
    num_problems = size(problems, 2);
    infos = cell(1, num_problems);


%     checkgradient(problems{2});
%     checkhessian(problems{2});
    
    for p = 1:num_problems
        problem = problems{p};
        fprintf('Solving %s\n', problem.name);
        [~, ~, info, ~] = steepestdescent(problem, X0s{p}, options);
        fprintf('\n');
        infos{p} = info;
    end

    for p = 1:num_problems
        problem = problems{p};
        fprintf('Solving %s\n', problem.name);
        [~, ~, info, ~] = trustregions(problem, X0s{p}, options);
        fprintf('\n');
        infos{p+num_problems} = info;
    end
    
    
    data_save_path = ['./data/', task]; 
    

    if ~exist(data_save_path, 'dir')
        mkdir(data_save_path);
    end

    infos_filename = fullfile(data_save_path, sprintf('infos_%.3f_%d.mat', ratio,r));
    save(infos_filename, 'infos');
    fprintf('Infos saved to: %s\n', infos_filename);
    
end

% record the test error
function stats = statsfun_test_error(problem, x, stats, store)
    [stats.cost_test,~] = problem.cost_test(x,store);
    [stats.cost_train,~] = problem.cost_train(x,store);
    fprintf("%d      %d",stats.cost_test,stats.cost_train);
end


%   Ex.1 Oblique(m,n)
function result = manifoldproj(h,u)
    % project the vector u in the embedding space onto T_hH
    % h,u\in R^{mxs}
    proj_alongh = (sum(h.*u,2)).*h;
    result = u - proj_alongh;
end

function result = manifoldprojH(u)
    % project the vector u in the embedding space onto H
    % h,u\in R^{mxs}
    row_norms = vecnorm(u, 2, 2);
    result = u ./ row_norms;
end
