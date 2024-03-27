module TensorFun

export tensor_vector_product, is_regular

using Oscar

import Oscar: charpoly, is_regular

function tensor_vector_product(A::Array{T, M}, x::Vector{T}) where {T, M}
  a = Base.tail(axes(A))
  @assert all(isequal(size(A, 1)), size(A))
  @assert length(x) === size(A, 1)
  R = parent(x[1])
  y = [zero(R) for i in 1:length(x)]
  for i in 1:size(A, 1)
    for ij in Iterators.product(a...)
      y[i] += A[i, ij...] * prod(x[collect(ij)])
    end
  end
  return y
end

function Oscar.charpoly(A::Array{T, M}) where {T, M}
  K = parent(first(A))
  return charpoly(K["t"][1], A)
end

function Oscar.charpoly(Kt, A::Array{T, M}) where {T, M}
  t = gen(Kt)
  n = size(A, 1) + 1
  Ktx, x = polynomial_ring(Kt, :x => 1:n)
  F = tensor_vector_product(Ktx.(A), x[1:n-1]) - t * x[n] .* x[1 : n - 1]
  push!(F, sum(x[1:n - 1].^2) - x[n]^2)
  return resultant(F)
end

function Oscar.is_regular(A::Array{T, M}) where {T <: Union{Integer, Rational}, M}
  return _is_regular(QQ.(A))
end

function Oscar.is_regular(A::Array{T, M}) where {T, M}
  return _is_regular(A)
end

function _is_regular(A::Array{T, M}) where {T, M}
  K = parent(first(A))
  n = size(A, 1)
  Kx, x = polynomial_ring(K, :x => 1:n)
  F = tensor_vector_product(Kx.(A), x[1:n])
  push!(F, sum(x[1:n].^2))
  return radical(ideal(F)) == ideal(x)
end

################################################################################
#
#  Classical resultant
#
################################################################################

# Classical resultant for families of polynomials. Inspired by
# "A package for computations with classical resultants"
# by Stagliano, http://dx.doi.org/10.2140/jsag.2018.8.21

function resultant(F::Vector{<: MPolyRingElem})
  # top-level resultant function, which does the argument verification
  @req length(F) > 0 "Resultant not defined for zero polynomials"
  R = parent(F[1])
  n = ngens(R) - 1
  @req n + 1 == length(F) "Number of polynomials ($(length(F))) must be number of variables ($(n + 1))"
  # Cannot test the following, because it is not implemented for all
  # rings
  # @req all(is_homogeneous, F) "Polynomials must be homogeneous)
  return _resultant(F)
end

function _resultant(F::Vector{<: MPolyRingElem{<: FieldElem}})
  # field case, apply Poisson formula directly
  return _resultant_poisson(F)
end

function _resultant(F::Vector{<: MPolyRingElem})
  # ring case, pass to the field of fractions if possible
  R = coefficient_ring(parent(F[1]))
  if is_domain_type(elem_type(R))
    K = fraction_field(R)
    Kx, x = polynomial_ring(K, ngens(parent(F[1])); cached = false)
    FK = map_coefficients.(Ref(K), F, cached = false, parent = Kx)
    r = _resultant_poisson(FK)
    @assert is_one(denominator(r))
    return numerator(r)
  else
    error("Resultant implemented only for domains. Please open an issue")
  end
end

function _resultant_poisson(F::Vector{<: MPolyRingElem{<:FieldElem}})
  # Resultant via Poisson formula apres
  # Theorem 3.4, p. 96 of Cox-Little-O'shea, Using Algebraic Geometry (2005)
  #
  # In case a naive application returns zero, it is inconclusive.
  # In this case we compute the dimension, followed by a random coordinate
  # transformation in the zero-dimensional case.

  R = parent(F[1])
  r = __resultant_poisson(F)
  if !is_zero(r)
    return r
  end

  # now r == 0, but we could have had bad projections
  d = dim(ideal(F))

  if d > 0
    # there are infinitely many solutions, hence a nontrivial one
    return r
  end

  # we make a random SL_n coordinate change, which does not change the resultant
  # See 1.5 (and the references) of "A package for computations with classical
  # resultants" by Stagliano, http://dx.doi.org/10.2140/jsag.2018.8.21
  
  # keep a counter to capture bugs
  k = 0
  while is_zero(r)
    k > 100 && error("Something wrong in the resultant. Please report this bug")
    h = _random_sln_transformation(R)
    r = __resultant_poisson(h.(F))
  end
  return r
end

function __resultant_poisson(F::Vector{<: MPolyRingElem{<:FieldElem}})
  # If we hit a zero, it is inconclusive and we return the zero
  #@req all(is_homogeneous, F) "Polynomials must be homogenous"
  R = parent(F[1])
  n = ngens(R) - 1
  K = coefficient_ring(R)
  d = total_degree.(F)
  # Could have more special cases
  if n == 0 # one polynomial
    return is_zero(F[1]) ? zero(K) : leading_coefficient(F[1])
  end
  S, x = polynomial_ring(K, :x => 0:n-1, cached = false)
  h = hom(R, S, vcat(x, [one(S)]))
  f = h.(F)
  h = hom(R, S, vcat(x, [zero(S)]))
  Fbar = h.(F)
  res0 = __resultant_poisson(Fbar[1:end-1])
  if is_zero(res0)
    return res0
  end
  I = ideal(S, f[1:end-1])
  if dim(I) > 0
    return zero(K)
  end
  Q, mQ = quo(S, I)
  V, VtoQ = vector_space(K, Q)
  @assert dim(V) == prod(d[1:end-1]; init = 1)
  M = zero_matrix(K, dim(V), dim(V))
  fn = mQ(last(f))
  for i in 1:dim(V)
    v = VtoQ\(VtoQ(V[i]) * fn)
    for j in 1:dim(V)
      M[i, j] = v[j]
    end
  end
  return res0^d[end] * det(M)
end

function _random_sln_transformation(R::MPolyRing)
  K = base_ring(R)
  n = ngens(R)
  @assert K isa Field
  A = identity_matrix(K, n)
  if n <= 1
    return A
  end
  # do five random elementary operations
  for l in 1:5
    i = rand(1:n)
    j = rand(1:n)
    while j == i
      j = rand(1:n)
    end
    add_row!(A, one(K), i, j)
  end
  X = gens(R)
  h = hom(R, R, [sum((A[i, j] * X[j] for j in 1:n); init = zero(R)) for i in 1:n], check = false)
  return h
end


end # module TensorFun
