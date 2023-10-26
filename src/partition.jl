#=
    Copyright (c) 2023 Louis Yu Lu, MIT License
=#

using LinearAlgebra, RandomizedLinAlg
using Statistics
using Logging

export
  Node,
  NodeTree,
  create_node_tree,
  populate_nodes!,
  find_leaf,
  Cell,
  Cluster,
  make_cluster!,
  populate_cells!,
  populate_neighbors!,
  closest_cell,
  top_match_cells,
  closest_sample,
  top_match_samples,
  top_matches

mutable struct Node{T<:AbstractFloat}
  indices::Vector{Int32}
  centroid::Vector{T}
  direction::Vector{T}
  level::Int8
  left::Int32
  right::Int32
end

function create_node(T::Type{<:AbstractFloat})::Node{T}
  Node(
    Int32[], T[], T[], Int8(0), Int32(0), Int32(0))
end

create_node() = create_node(Float32)

function calc_dir_proj(ds::Matrix{T})::Tuple{Vector{T},Vector{T}} where {T<:AbstractFloat}
  # farthest point for the direction
  # n1, n2 = size(ds)
  # dir = zeros(T, n1)
  # max_d = T(0)
  # for i in 1:n2
  #   d = norm(ds[:, i])
  #   if d > max_d
  #     max_d = d
  #     dir = ds[:, i]
  #   end
  # end
  # dir /= max_d

  # longest axis direction
  u, s, v = rsvd(ds, 1)
  dir = convert(Matrix{T}, u)
  proj = convert(Matrix{T}, s .* v')
  vec(dir), vec(proj)
end

mutable struct NodeTree{T<:AbstractFloat}
  samples::Union{Matrix{T},Nothing}
  indices::Vector{Int32}
  tree::Vector{Node{T}}
  leaves::Vector{Int32}
  unprocessed::Vector{Node{T}}
end

function create_node_tree(samples::Matrix{T})::NodeTree{T} where {T<:AbstractFloat}
  NodeTree{T}(
    samples,
    Int32[],
    Node{T}[],
    Int32[],
    Node{T}[])
end

function split_node!(t::NodeTree{T}, indices::Vector{Int32})::Tuple{Vector{Int32},Vector{Int32},Vector{T},Vector{T}} where {T<:AbstractFloat}
  subset = Matrix{T}(undef, size(t.samples, 1), length(indices))
  for (k, i) in enumerate(indices)
    subset[:, k] = t.samples[:, i]
  end
  centroid = mean(subset, dims=2)
  ds = subset .- centroid
  direction, projections = calc_dir_proj(ds)
  left_indices = Int32[]
  right_indices = Int32[]
  for (i, idx) in enumerate(indices)
    if projections[i] <= 0.0
      push!(left_indices, idx)
    else
      push!(right_indices, idx)
    end
  end
  left_indices, right_indices, vec(centroid), direction
end

function split_root!(t::NodeTree{T})::Tuple{Vector{Int32},Vector{Int32},Vector{T},Vector{T}} where {T<:AbstractFloat}
  centroid = mean(t.samples, dims=2)
  ds = t.samples .- centroid
  direction, projections = calc_dir_proj(ds)
  left_indices = Int32[]
  right_indices = Int32[]
  m = size(t.samples)[2]
  for idx in 1:m
    if projections[idx] <= 0.0
      push!(left_indices, idx)
    else
      push!(right_indices, idx)
    end
  end
  left_indices, right_indices, vec(centroid), direction
end

function populate_nodes!(t::NodeTree{T}, max_level::Int, min_samples::Int) where {T<:AbstractFloat}
  leftIx, rightIx, ct, dr = length(t.indices) == 0 ? split_root!(t) : split_node!(t, t.indices)
  dim = size(t.samples)[1]

  root = create_node(T)
  root.centroid = ct
  root.direction = dr
  root.level = 1
  push!(t.tree, root)

  left = create_node(T)
  left.level = 2
  left.indices = leftIx
  root.left = length(t.tree)
  push!(t.tree, left)
  push!(t.unprocessed, left)

  right = create_node(T)
  right.level = 2
  right.indices = rightIx
  root.right = length(t.tree)
  push!(t.tree, right)
  push!(t.unprocessed, right)
  @info "populate_nodes! left_cnt=$(length(leftIx)) right_cnt=$(length(rightIx))"
  while length(t.unprocessed) > 0
    nd = popfirst!(t.unprocessed)
    # @debug "level=$(nd.level) cnt=$(length(nd.indices))"
    leftIx, rightIx, ct, dr = split_node!(t, nd.indices)
    nd.centroid = ct
    nd.direction = dr
    if length(leftIx) == 0 || length(rightIx) == 0
      continue
    end
    if nd.level >= max_level || length(nd.indices) <= min_samples
      continue
    end
    left = create_node(T)
    left.level = nd.level + 1
    left.indices = leftIx
    nd.left = length(t.tree)
    push!(t.tree, left)
    push!(t.unprocessed, left)

    right = create_node(T)
    right.level = nd.level + 1
    right.indices = rightIx
    nd.right = length(t.tree)
    push!(t.tree, right)
    push!(t.unprocessed, right)

    resize!(nd.indices, 0) # clear the memory usage
    # @debug "level=$(nd.level), left_cnt=$(length(leftIx)), right_cnt=$(length(rightIx))"
  end
  for (i, nd) in enumerate(t.tree)
    if nd.left == 0 && nd.right == 0
      push!(t.leaves, i)
    end
  end
end

function find_leaf(t::NodeTree{T}, x::Vector{T})::Int32 where {T<:AbstractFloat}
  k::Int32 = 1 # start from the root
  nd = t.tree[k]
  while true
    if nd.left == 0 && nd.right == 0
      break
    end
    proj = sum((x - nd.centroid) .* nd.direction)
    # @debug "x: $(size(x)), nd.direction: $(size(nd.direction)), proj: $(size(proj))"
    k = proj <= 0 ? nd.left : nd.right
  end
  k
end

mutable struct Cell{T<:AbstractFloat}
  centroid::Vector{T}
  r2max::T
  indices::Vector{Int32}
  neighbors::Vector{Int32}
end

function create_cell(nd::Node{T})::Cell{T} where {T<:AbstractFloat}
  Cell(nd.centroid, T(0), Int32[], Int32[])
end

mutable struct Cluster{T<:AbstractFloat}
  samples::Matrix{T}
  indices::Vector{Int32}
  cells::Vector{Cell{T}}
end

function make_cluster!(t::NodeTree{T})::Cluster{T} where {T<:AbstractFloat}
  cluster = Cluster{T}(t.samples, t.indices, Cell{T}[])
  t.samples = nothing
  for k in t.leaves
    nd = t.tree[k]
    push!(cluster.cells, create_cell(nd))
  end
  cluster
end

function populate_cells!(cluster::Cluster{T}) where {T<:AbstractFloat}
  dim, m = size(cluster.samples)
  n = length(cluster.cells)
  cs = Matrix{T}(undef, dim, n)
  for i in 1:n
    cs[:, i] = cluster.cells[i].centroid
  end
  m2 = length(cluster.indices)
  @info "populate_cells! count: $(m2 == 0 ? m : m2)"
  for k in (m2 == 0 ? (1:m) : cluster.indices)
    r2 = vec(sum((cs .- cluster.samples[:, k]) .^ 2, dims=1))
    k_min = sortperm(r2)[1]
    r2_min = r2[k_min]
    c = cluster.cells[k_min]
    push!(c.indices, k)
    if r2_min > c.r2max
      c.r2max = r2_min
    end
  end
  # delete empty cells
  deleteat!(cluster.cells, findall(c -> length(c.indices) == 0, cluster.cells))
end

function populate_neighbors!(cluster::Cluster{T}) where {T<:AbstractFloat}
  dim = size(cluster.samples)[1]
  n = length(cluster.cells)
  cs = Matrix{T}(undef, dim, n)
  @info "populate_neighbors! cells: $n"
  for i in 1:n
    cs[:, i] = cluster.cells[i].centroid
  end
  for i in 1:n
    for j in (i+1):n
      a = (cs[:, i] .+ cs[:, j]) / T(2)
      r2 = vec(sum((cs .- a) .^ 2, dims=1))
      idx_r2 = sortperm(r2)
      k1 = idx_r2[1]
      d1 = r2[k1]
      k2 = idx_r2[2]
      d2 = r2[k2]
      if (k1 == i && k2 == j) || (k2 == i && k1 == j) &&
                                 d1 <= cluster.cells[k1].r2max &&
                                 d2 <= cluster.cells[k2].r2max
        push!(cluster.cells[i].neighbors, j)
        push!(cluster.cells[j].neighbors, i)
      end
    end
  end
end

function closest_cell(cluster::Cluster{T}, x::Vector{T})::Tuple{Int,Cell{T}} where {T<:AbstractFloat}
  xdims = length(x)
  n = length(cluster.cells)
  d2 = Vector{T}(undef, n)
  for i in 1:n
    dx = x - cluster.cells[i].centroid[1:xdims]
    d2[i] = sum(dx .^ 2)
  end
  idx = sortperm(d2)[1]
  idx, cluster.cells[idx]
end

function top_match_cells(cluster::Cluster{T}, x::Vector{T}, n_top::Int)::Tuple{Vector{Int},Vector{Cell{T}}} where {T<:AbstractFloat}
  xdims = length(x)
  n = length(cluster.cells)
  d2 = Vector{T}(undef, n)
  for i in 1:n
    dx = x - cluster.cells[i].centroid[1:xdims]
    d2[i] = sum(dx .^ 2)
  end
  ks = sortperm(d2)[1:n_top]
  ks, d2[ks]
end

function closest_sample(cluster::Cluster{T}, x::Vector{T}, k::Int)::Tuple{Int,Vector{T},T} where {T<:AbstractFloat}
  xdims = length(x)
  c = cluster.cells[k]
  top = c.indices[1]
  r = cluster.samples[:, c.indices[1]]
  ds_min = sum((r[1:xdims] - x) .^ 2)
  for j in 2:length(c.indices)
    i = c.indices[j]
    z = cluster.samples[:, i]
    ds = sum((z[1:xdims] - x) .^ 2)
    if ds < ds_min
      ds_min = ds
      top = i
      r = z
    end
  end
  top, r, ds_min
end

function top_match_samples(cluster::Cluster{T}, x::Vector{T}, c::Cell{T}, n_top)::Tuple{Vector{Int},Vector{T}} where {T<:AbstractFloat}
  xdims = length(x)
  xs = cluster.samples[1:xdims, c.indices]
  d2 = vec(sum((x .- xs) .^ 2, dims=1))
  ks = sortperm(d2)[1:n_top]
  c.indices[ks], d2[ks]
end

function top_matches(cluster::Cluster{T}, x::Vector{T}, ntop::Int)::Tuple{Vector{Int},Matrix{T},Vector{T}} where {T<:AbstractFloat}
  xdims = length(x)
  m = length(cluster.cells)
  xs = Matrix{T}(undef, xdims, m)
  for i in 1:m
    xs[:, i] = cluster.cells[i].centroid[1:xdims]
  end
  d2 = vec(sum((x .- xs) .^ 2, dims=1))
  ks = sortperm(d2)[1:ntop]
  tis = zeros(Int, ntop)
  tzs = zeros(T, size(cluster.samples, 1), ntop)
  tds = zeros(T, ntop)
  for (i, k) in enumerate(ks)
    tis[i], tzs[:, i], tds[i] = closest_sample(cluster, x, k)
  end
  tis, tzs, tds
end