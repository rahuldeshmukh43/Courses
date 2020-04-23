function [x_star, history_out] = particleswarm(fun,a,b, Nswarm, Niters,...
    inert_const, cog_const,social_const, constricted, vmax_prop)
%   a,b are the limits of the feasible domains of x i.e. x \in (a,b)
    history.name = 'Global Optimizer: PSO';
%     rng('default');
    switch nargin
        case 3
            Nswarm = 40;
            Niters = 100;
            constricted = 1;
            inert_const = 0.8;
            cog_const = 2;
            social_const = 2;
            vmax_prop = 0.1;
        case 4
            Niters = 100;
            constricted = 1;
            inert_const = 0.8;
            cog_const = 2;
            social_const = 2;
            vmax_prop = 0.1;
        case 5
            inert_const = 0.8;
            cog_const = 2;
            social_const = 2;
            constricted = 1;
            vmax_prop = 0.1;
        case 6
            cog_const = 2;
            social_const = 2;
            constricted = 1;
            vmax_prop = 0.1;
        case 7
            social_const = 2;
            constricted = 1;
            vmax_prop = 0.1;
        case 8
            constricted = 1;
            vmax_prop = 0.1;
        case 9 
            vmax_prop = 0.1;
    end
    x_dim = length(a);
    vmax = vmax_prop*(b-a);
    history.parameter.x_dim= x_dim;
    history.parameter.Nswarm = Nswarm;
    history.parameter.Niters = Niters;
    history.parameter.inert_const = inert_const;
    history.parameter.cog_const = cog_const;
    history.parameter.social_const = social_const;
    history.parameter.constricted = constricted;
    history.parameter.vmax = vmax;
    if constricted
       phi = cog_const + social_const;
       kappa = 2/abs(2-phi -sqrt(phi^2 -4*phi));
    end
    
    count = 0;
    % generate the swarm randomly
    X_swarm = rand(x_dim, Nswarm); % positions \in (0,1)
    V_swarm = 2*rand(x_dim, Nswarm)-1; % velocities \in (-1,1)
    V_swarm =  min(vmax,max(-vmax,V_swarm)); % \in (-vmax,vmax)
    % scale to the domain
    X_swarm = (b-a).*X_swarm + a; 
    % update pbest and gbest
    pbest_x = X_swarm;
    pbest_fval = fun(X_swarm);
    [gbest_fval, idx] = min(pbest_fval);
    gbest_x = pbest_x(:,idx);
    % write to history
    history.data(count+1).pbest_fval = pbest_fval;
    history.data(count+1).gbest_x = gbest_x;
    history.data(count+1).gbest_fval = gbest_fval;
    history.data(count+1).pbest_av = mean(pbest_fval);
    history.data(count+1).pbest_worse = max(pbest_fval);
    
    for count=1:Niters
        % generate r and s
        r = rand(x_dim,Nswarm);
        s = rand(x_dim,Nswarm);
        % update velocity
        V_swarm = inert_const*V_swarm + cog_const*(r.*(pbest_x-X_swarm)) + ...
                  social_const*(s.*(gbest_x-X_swarm));
        if constricted
           V_swarm = kappa*V_swarm;
        end
        % clamp velocities
        V_swarm =  min(vmax,max(-vmax,V_swarm)); % \in (-vmax,vmax)
        %update position
        X_swarm = X_swarm + V_swarm;
        %update pbest
        new_fval = fun(X_swarm);
        for i=1:Nswarm
           if new_fval(i)< pbest_fval(i)
              pbest_fval(i) = new_fval(i);
              pbest_x(:,i) = X_swarm(:,i);
           end
        end 
        %update gbest
        if sum(pbest_fval < gbest_fval) > 0
           [gbest_fval,idx] = min(pbest_fval); 
           gbest_x = X_swarm(:,idx);
        end
        % write to history
        history.data(count+1).pbest_fval = pbest_fval;
        history.data(count+1).gbest_x = gbest_x;
        history.data(count+1).gbest_fval = gbest_fval;
        history.data(count+1).pbest_av = mean(pbest_fval);
        history.data(count+1).pbest_worse = max(pbest_fval);        
    end
    history.Niters = count;
    x_star = gbest_x;
    if nargout>1
        history_out = history;
    end  
end
