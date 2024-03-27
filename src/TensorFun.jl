module TensorFun

export tensor_vector_product, is_regular, charpoly

using Oscar

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

function charpoly(A::Array{T, M}) where {T, M}
  K = parent(first(A))
  return charpoly(K["t"][1], A)
end

function charpoly(Kt, A::Array{T, M}) where {T, M}
  t = gen(Kt)
  n = size(A, 1) + 1
  Ktx, x = polynomial_ring(Kt, :x => 1:n)
  F = tensor_vector_product(Ktx.(A), x[1:n-1]) - t * x[n] .* x[1 : n - 1]
  push!(F, sum(x[1:n - 1].^2) - x[n]^2)
  return resultant(F)
end

function is_regular(A::Array{T, M}) where {T <: Union{Integer, Rational}, M}
  return _is_regular(QQ.(A))
end

function is_regular(A::Array{T, M}) where {T, M}
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

end # module TensorFun
