#=
    Copyright (c) 2023 Louis Yu Lu, MIT License
=#

export
    word_ngram,
    vecs_cluster,
    time_span,
    cos_fn,
    cos_1d_mask,
    cos_2d_mask

function vecs_cluster(xs::Matrix{Float32}, max_level::Int, min_samples::Int)::Cluster{Float32}
    nt = create_node_tree(xs)
    populate_nodes!(nt, max_level, min_samples)
    cluster = make_cluster!(nt)
    populate_cells!(cluster)
    @info "cells:", size(cluster.cells)
    cluster
end

function time_span(from::Dates.DateTime, to::Dates.DateTime, sec_min::Symbol)::String
    dt = to - from
    if sec_min == :Second
        "$(round(dt.value / 1_000, digits=4)) seconds"
    elseif sec_min == :Minute
        "$(round(dt.value / 60_000, digits=4)) minutes"
    else
        ""
    end
end

function cos_fn(N::Int, m::Int, k::Int)
    cos(pi * (N - m) * (k - 1) / (N - 1))
end

function cos_1d_mask(N::Int, m::Int, k::Int)::Matrix{Float32}
    mask = Matrix{Float32}(undef, k, m)
    for i in 1:m
        for j in 1:k
            mask[j, i] = cos_fn(N, i, j)
        end
    end
    mask
end

function cos_2d_mask(N::Int, m::Int, k::Int)::Array{Float32}
    mask1 = cos_1d_mask(N, m, k)
    n1, n2 = size(mask1)
    mask2 = Array{Float32}(undef, n1, n1, n2, n2)
    for i in 1:n2
        for j in 1:n2
            mask2[:, :, j, i] = mask1[:, j] * mask1[:, i]'
        end
    end
    mask2
end

