module dynopt_rk

    export ButcherTableau, RK4, RungeKuttaTape, rungeKutta, rungeKuttaTraj, rungeKuttaTaping, 
           rungeKuttaAdjoint

    ## A small julia collection of routines for dynamic optimization

    struct ButcherTableau
        a::Matrix{Float64}
        b1::Vector{Float64}
        b2::Vector{Float64}
        c::Vector{Float64}
        o::Int64
    end

    RK4 = ButcherTableau(
        [  0    0  0  0
        1/2    0  0  0
        0  1/2  0  0
        0    0  1  0],
        [1/6, 1/3, 1/3, 1/6],
        [0.0],
        [0, 1/2, 1/2, 1],
        4
    );


  # A fixed-stepsize Runge Kutta method, given a Tableau, that solves the initial value problem
    # x' = f(t,x) on t in [t0,tf]
    function rungeKutta(
        f::Function, x0::Vector, t0::Float64, tf::Float64, h::Float64, BT::ButcherTableau)
        # dimensions and allocation of variables
        dim_s = size(BT.c,1)
        dim_x = size(x0,1)
        k = zeros(typeof(x0[1]), dim_s, dim_x)
        # initial values
        t = t0
        eta = x0
        # book keeping
        n_evals = 0
        n_steps = 1
        # the time stepping loop starts here
        while abs(tf-t) > 1e-4*abs(h)
            # step size h, but step onto final time tf if reachable
            h = (tf>t) ? min(tf-t, h) : max(tf-t, h)
            # evaluate s stages k_j = f(t + h*c_j, y_j)
            #                in y_j = eta + h*sum(a_jl, k_l over l = 1 .. j-1)
            for jj in 1:dim_s
                y_jj = eta + h * (transpose(k[1:jj-1,:]) * BT.a[jj,1:jj-1])
                k[jj,:] = f(t + h*BT.c[jj], y_jj)
            end
            # advance eta and t
            eta += h * (transpose(k) * BT.b1)
            t += h
            # book keeping
            n_steps += 1
            n_evals += dim_s
        end
        return eta, n_steps, n_evals
    end


    # A fixed-stepsize Runge Kutta method, given a Tableau, that solves the initial value problem
    # x' = f(t,x) on t in [t0,tf]
    function rungeKuttaTraj(
        f::Function, x0::Vector, t0::Float64, tf::Float64, h::Float64, BT::ButcherTableau)
        # dimensions and allocation of variables
        dim_s = size(BT.c,1)
        dim_x = size(x0,1)
        k = zeros(typeof(x0[1]), dim_s, dim_x)
        # initial values
        t = t0
        eta = x0
        # book keeping
        n_steps = Int64(2+ceil((tf-t0)/h))
        n_evals = 0
        x_traj = zeros(typeof(x0[1]), dim_x, n_steps)
        t_traj = zeros(n_steps)
        x_traj[:,1] = eta;
        t_traj[1] = t;
        n_steps = 1
        # the time stepping loop starts here
        while abs(tf-t) > 1e-4*abs(h)
            # step size h, but step onto final time tf if reachable
            h = (tf>t) ? min(tf-t, h) : max(tf-t, h)
            # evaluate s stages k_j = f(t + h*c_j, y_j)
            #                in y_j = eta + h*sum(a_jl, k_l over l = 1 .. j-1)
            for jj in 1:dim_s
                y_jj = eta + h * (transpose(k[1:jj-1,:]) * BT.a[jj,1:jj-1])
                k[jj,:] = f(t + h*BT.c[jj], y_jj)
            end
            # advance eta and t
            eta += h * (transpose(k) * BT.b1)
            t += h
            # book keeping
            n_steps += 1
            n_evals += dim_s
            x_traj[:,n_steps] = eta      # careful here: don't just append, but preallocate everything!
            t_traj[n_steps] = t
        end
        x_traj = x_traj[:,1:n_steps]
        t_traj = t_traj[1:n_steps]
        return t_traj, x_traj, n_steps, n_evals
    end


    mutable struct RungeKuttaTape
        t0 :: Float64
        tf :: Float64
        BT :: ButcherTableau
        t  :: Vector{Float64}     # [step]
        h  :: Vector{Float64}     # [step]
        y  :: Array{Float64, 3}   # [step,stage,nx]
    
        RungeKuttaTape(t0, tf, BT, n_steps, nx) =
            new(t0, tf, BT, zeros(Float64, n_steps), zeros(Float64, n_steps), zeros(Float64, n_steps, size(BT.c,1), nx))
    end

    function rungeKuttaTaping(
        f::Function, x0::Vector, t0::Float64, tf::Float64, h::Float64, BT::ButcherTableau) :: RungeKuttaTape
        # dimensions and allocation of variables
        dim_s = size(BT.c,1)
        dim_x = size(x0,1)
        k = zeros(typeof(x0[1]), dim_s, dim_x)
        # initial values
        t = t0
        eta = x0
        # book keeping
        n_steps = Int64(2+ceil((tf-t0)/h))
        tape = RungeKuttaTape(t0, tf, BT, n_steps, dim_x)
        n_steps = 1
        # the time stepping loop starts here
        while abs(tf-t) > 1e-4*abs(h)
            tape.t[n_steps] = t
            h = (tf>t) ? min(tf-t, h) : max(tf-t, h)
            tape.h[n_steps] = h
            for jj in 1:dim_s
                # write stage evaluation points y_j to tape
                tape.y[n_steps,jj,:] = eta + h * (k[1:jj-1,:]' * BT.a[jj,1:jj-1])   
                k[jj,:] = f(t + h*BT.c[jj], tape.y[n_steps,jj,:] )
            end
            eta += h * (k' * BT.b1)
            t += h
            n_steps += 1
        end
        # write final time and result to tape
        tape.t[n_steps] = t
        tape.y[n_steps,1,:] = eta
        tape.t = tape.t[1:n_steps]
        tape.y = tape.y[1:n_steps,:,:]
        return tape
    end



    function rungeKuttaAdjoint(
        fadj::Function, lamf::Vector{Float64}, tape::RungeKuttaTape)
        # dimensions and allocation of variables
        dim_s = size(tape.BT.c,1)
        dim_x = size(lamf,1)
        theta = zeros(dim_s, dim_x)
        # final values
        lam = lamf
        # the backward propagation time stepping loop starts here
        for k = size(tape.t,1)-1:-1:1
            # read time and step size from tape
            h = tape.h[k]    # do not compute by subtracting t timestamps! Truncation errors!
            t = tape.t[k]
            # start propagation with incoming lambda
            lam_old = copy(lam)
            for ii in dim_s:-1:1
                # compute adjoint direction for lam'*df/dx
                lam_ii = tape.BT.b1[ii]*lam_old + theta[ii+1:dim_s,:]' * tape.BT.a[ii+1:dim_s,ii] 
                theta[ii,:] = h * fadj(t + h * tape.BT.c[ii], tape.y[k,ii,:], lam_ii)
                # update outgoing lambda
                lam += theta[ii,:]
            end
        end
        return lam
    end    
end
