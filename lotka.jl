module lotka

    export IVPStruct, LotkaIVP, BVPStruct, LotkaBVP

    #
    # An abstract type for an initial value problem
    #
    mutable struct IVPStruct
        nx    :: Integer
        np    :: Integer
        f     :: Function
        dfdx  :: Function
        dfdp  :: Function
        fadj  :: Function
    end

    function LotkaIVP()
        return IVPStruct(2, 2,  
            lotka_f,
            lotka_dfdx,
            lotka_dfdp,
            lotka_fadj
        )
    end

    #
    # An abstract type for a boundary value problem
    #
    mutable struct BVPStruct
        ivp   :: IVPStruct
        nr    :: Integer
        r     :: Function
        drdx0 :: Function
        drdx1 :: Function
        drdp  :: Function
    end



    function lotka_f(t::Float64, x::Vector, p::Vector{Float64})
        return [
            x[1] * (p[1] - x[2])
            -x[2] * (p[2] - x[1])
        ]
    end

    # df/dx
    function lotka_dfdx(t::Float64, x::Vector{Float64}, p::Vector{Float64})
        return [
            p[1]-x[2]    -x[1]
            x[2]    -(p[2]-x[1])
        ]
    end

    # df/dp
    function lotka_dfdp(t::Float64, x::Vector{Float64}, p::Vector{Float64})
        return [
            x[1]    0.0
            0.0     -x[2]
        ]
    end

    # df/dx'*lam
    function lotka_fadj(t::Float64, x::Vector{Float64}, lam::Vector{Float64}, p::Vector{Float64})
        return [
            lam[1]*(p[1]-x[2]) + lam[2]*x[2],
            - lam[1]*x[1] - lam[2]*(p[2]-x[1]) 
        ]
    end

    function lotka_r(x0::Vector, x1::Vector, p::Vector{Float64})
        return [
            x0[1] - 1,
            x0[2] - 1,
            x1[1] - 2,
            x1[2] - 0.5
        ]
    end

    function lotka_drdx0(x0::Vector, x1::Vector, p::Vector{Float64})
        return [
            1 0 
            0 1 
            0 0 
            0 0 
        ]
    end

    function lotka_drdx1(x0::Vector, x1::Vector, p::Vector{Float64})
        return [
            0 0
            0 0
            1 0
            0 1
        ]
    end

    function lotka_drdp(x0::Vector, x1::Vector, p::Vector{Float64})
        return [
            0 0
            0 0
            0 0
            0 0
        ]
    end

    function LotkaBVP()
            return BVPStruct(
                # the underlying IVP
                IVPStruct(2, 2, 
                    lotka_f,
                    lotka_dfdx,
                    lotka_dfdp,
                    lotka_fadj
                ),
                # the BVP ingredients
                4,
                lotka_r,
                lotka_drdx0,
                lotka_drdx1,
                lotka_drdp
            )
    end

end
