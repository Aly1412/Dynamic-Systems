module dynopt_newton

using Printf, LinearAlgebra

export newton, NewtonFunction, newton_eval_funder, newton_eval_fun, newton_solve

#
# the abstract type of a function of which newton() can find roots
# Concrete types must at least implement the functions
# - newton_eval_fun(F :: NewtonFunction, x :: Vector) :: Vector
# - newton_eval_funder(F :: NewtonFunction, x :: Vector) :: Tuple{Real, Matrix}
# - newton_solve(F :: NewtonFunction, D :: Matrix, x :: Vector) :: Vector
abstract type NewtonFunction end

function newton_eval_funder(
    F :: NewtonFunction,
    xk :: Vector
) :: Tuple{Vector, Matrix}
end

function newton_eval_fun(
    F :: NewtonFunction,
    xk :: Vector
) :: Vector
end

function newton_solve(
    F :: NewtonFunction,
    D :: Matrix,
    x :: Vector
) :: Vector
end

function newton_callback(
    F :: NewtonFunction,
    k :: Integer,
    xk :: Vector,
    Fnrm :: Real,
    status :: Integer
)
end

function newton(
    F :: NewtonFunction,               # F(x,der) returns F(x), and dF(x)/dx if der==true
    x0 :: Vector{Float64},             # The initial guess
    itmax :: Int64 = 100,              # The maximum iteration count
    tol :: Float64 = 1.0e-8,           # The acceptable 2-norm residual
    alpha_min :: Float64 = 1.0e-12,    # The minimum step size before we bail out
    gamma :: Float64 = 0.2,            # Armijo's tuning parameter
    print :: Int64 = 2                 # The print level (0=errors, 1=successes, 2=iterations)
)
    xk = copy(x0)
    
    # Newton iteration loop
    if print >=2
        @printf("%4s %12s  %12s  %12s  %s\n", "iter", "||resid||", "||step||", "size", "type")
    end
    newton_callback(F, 0, xk, 0, 0)
    for k = 1:itmax
        # evaluate F(x^k) and its derivative dF(x^k)/dx
        Fk, dFdxk = newton_eval_funder(F, xk)

        if any(isnan.(Fk)) || any(.!isfinite.(Fk))
            if print >= 0
                println("Function evaluation failed.")
            end
            return xk
        end

        # termination criterion ||F(x^k)|| < tol satisfied?
        Fknrm = norm(Fk, 2)
        if Fknrm < tol 
            if print >= 2
                @printf("%3d  %12.4e\n", k, Fknrm)
            end
            if print >= 1
                println("Root finding problem solved to precision $Fknrm.")
            end
            newton_callback(F, k, xk, Fknrm, 1)
            return xk
        end

        # Compute the Newton direction deltak = - dFdxk^(-1) * Fk; fall back to a gradient direction  
        local deltak
        flag = " "
        try
            deltak = - newton_solve(F, dFdxk, Fk)
            flag = "N"  # N = Newton step
        catch e
            if print >= 0
                println("Exception caught:", e)
            end
            deltak = - Fk
            flag = "G"  # G = gradient descent step
        end

        # Try line search on the l2 merit function phi(a) = 0.5*||F(x+a*d)||_2^2
        # The gradient in a=0 is phi'(0) = dF/dx(x)^T * F(x)
        # The slope into the Newton direction is phi'(0)^T * delta^k = -F(x^k)^T * F(x^k) = -Fknrm
        # Evaluate F without its derivative for efficiency.
        alpha = 1.0
        alpha_min = 1.0e-12
        while (alpha > alpha_min) && (0.5*norm(newton_eval_fun(F, xk+alpha*deltak),2) >= 0.5*Fknrm - gamma * alpha * Fknrm)
            alpha /= 2.0
        end

        # Take step
        xk += alpha * deltak

        if print >= 2
            @printf("%3d  %12.4e  %12.4e  %12.4e  %s\n", k, Fknrm, norm(deltak, 2), alpha, flag)
        end
        newton_callback(F, k, xk, Fknrm, 1)

        # Line search stalled?
        if alpha <= alpha_min
            if print >= 0
                println("Line search stalled!")
            end
            return xk
        end
    end
    if print >= 0
        println("Maximum Newton iteration count $itmax reached.")
    end
    return xk
end

end
