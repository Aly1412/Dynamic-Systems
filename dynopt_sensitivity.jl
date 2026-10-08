module dynopt_sensitivity

using LinearAlgebra

export augmentComplexStepRhs, augmentComplexStepIv
export augmentRealStepRhs, augmentRealStepIv
export augmentStateVariationRhs, augmentStateVariationIv
export augmentParamVariationRhs, augmentParamVariationIv
export augmentBothVariationRhs, augmentBothVariationIv

#
# Augment an nx-dimensional initial value by
# nx copies with perturbed imaginary components.
#
function augmentComplexStepIv(
    x0 :: Vector{Float64},
    pert :: Float64
)
    nx = size(x0,1)
    return vcat(
        x0, 
        [
            x0 + im*pert*(1:nx .== i) 
            for i in 1:nx
        ]...
    )
end


#
# Augment a complex differentiable right hand side f(x)
# by nx copies of itself.
#
function augmentComplexStepRhs(
    f :: Function,
    x :: Vector{ComplexF64},
    nx :: Int64
)
    return vcat(
        [ 
            f(x[ i*nx+1 : (i+1)*nx ])  
            for i in 0:nx 
        ]...
    )
end


#
# Augment an nx-dimensional initial value by
# nx copies with perturbed real components.
#
function augmentRealStepIv(
    x0 :: Vector{Float64},
    pert :: Float64
)
    nx = size(x0,1)
    return vcat(
        x0, 
        [
            x0 + pert*(1:nx .== i) 
            for i in 1:nx
        ]...
    )
end


#
# Augment a real differentiable right hand side f(x)
# by nx copies of itself.
#
function augmentRealStepRhs(
    f :: Function,
    x :: Vector{Float64},
    nx :: Int64
)
    return vcat(
        [ 
            f(x[ i*nx+1 : (i+1)*nx ])  
            for i in 0:nx 
        ]...
    )
end


#
# Augment an nx-dimensional initial value by
# the columns of the nx-by-nx identity matrix
#
function augmentStateVariationIv(
    x0 :: Vector{Float64}
)
    nx = size(x0,1)
    return vcat(
        x0, 
        reshape(Matrix(I, nx, nx), nx*nx)
    )
end


#
# Augment a real differentiable right hand side f(x)
# by the variational equation
#   df/dx * W
# where the columns of W are expected to follow x in 
# the state vector
#
function augmentStateVariationRhs(
    f :: Function,
    dfdx :: Function,
    x :: Vector{Float64},
    nx :: Int64
)
    return vcat(
        [ 
            f(x[1:nx]),
            reshape(
                dfdx(x[1:nx]) * reshape(x[nx+1:nx*(nx+1)],
                                        (nx,nx)), 
                nx*nx)
        ]...
    )
end


#
# Augment an nx-dimensional initial value by
# the columns of the nx-by-np zero matrix
#
function augmentParamVariationIv(
    x0 :: Vector{Float64},
    np :: Int64
)
    nx = size(x0,1)
    return vcat(
        x0, 
        zeros(nx*np)
    )
end


#
# Augment a real differentiable right hand side f(x)
# by the variational equation
#   df/dx * Wp + df/dp
# where the columns of Wp are expected to follow x in 
# the state vector
#
function augmentParamVariationRhs(
    f :: Function,
    dfdx :: Function,
    dfdp :: Function,
    x :: Vector{Float64},
    nx :: Int64,
    np :: Int64
)
    return vcat(
        [ 
            f(x[1:nx]),
            reshape(
                dfdx(x[1:nx]) * reshape(x[nx+1:nx*(nx+1)],
                                        (nx,np))
                + dfdp(x[1:nx]), 
                nx*np)
        ]...
    )
end



#
# Augment an nx-dimensional initial value by
# the columns of the nx-by-(nx+np) matrix [I 0]
#
function augmentBothVariationIv(
    x0 :: Vector{Float64},
    np :: Int64
)
    nx = size(x0,1)
    return vcat(
        x0, 
        reshape(Matrix(I, nx, nx), nx*nx),
        zeros(nx*np)
    )
end


#
# Augment a real differentiable right hand side f(x)
# by the variational equations
#   df/dx * W,
#   df/dx * Wp + df/dp
# where the columns of W and Wp are expected to follow x 
# in the state vector
#
function augmentBothVariationRhs(
    f :: Function,
    dfdx :: Function,
    dfdp :: Function,
    x :: Vector{Float64},
    nx :: Int64,
    np :: Int64
)
    return vcat(
        [ 
            f(x[1:nx]),
            reshape(
                dfdx(x[1:nx]) * reshape(x[nx+1:nx*(nx+1)],
                                        (nx,nx)), 
                nx*nx),
            reshape(
                dfdx(x[1:nx]) * reshape(x[nx*(nx+1)+1:nx*(nx+np+1)],
                                        (nx,np))
                + dfdp(x[1:nx]), 
                nx*np)
        ]...
    )
end

end
