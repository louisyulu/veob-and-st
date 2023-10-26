### A Pluto.jl notebook ###
# v0.19.29

#=
    Copyright (c) 2023 Louis Yu Lu, MIT License
=#

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ e09d6b44-6370-11ed-1d3e-3fa29ee591a3
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate("../Project.toml")
end

# ╔═╡ 7d65c9d5-6dd7-45ac-8aee-b0b1d94329b9
using PlutoUI, Plots, Logging, Dates, LinearAlgebra, RandomizedLinAlg, Statistics, SparseArrays,
    Serialization, HypertextLiteral

# ╔═╡ 0ead4006-0a2d-4eb5-bac9-23077729e79b
begin
    include("../src/partition.jl")
    include("../src/my_utils.jl")
end

# ╔═╡ 76fbeff9-3a2b-48db-bd5e-fb5afce8e701
# Make Dispay Area Wider
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	 padding-left: max(50px, 5%);
    	 padding-right: max(50px, 10%);
	}
</style>
"""

# ╔═╡ 3bb7c7a8-d8cf-4f39-ab31-f7d670ad3b49
begin
    logger = ConsoleLogger(stdout, Logging.Info)
    global_logger(logger)
end

# ╔═╡ e7549436-2c9d-4edb-88aa-3e443c7be581
begin
    words, word_idx, sents, sent_src = deserialize("../data/imdb_words_sents.jls")
    n_words = length(words)
    @info n_words, words, word_idx, length(sents), sents, sent_src
end

# ╔═╡ 430c58e1-b25c-4fa3-91e2-8b77b62518a7
function word_ctx2_add_sent(w_before::Vector{Dict{Int,Float32}}, w_after::Vector{Dict{Int,Float32}}, words_sent::Vector{String}, 	word_idx::Dict{String,Int64})
    tkns = [word_idx[w] for w in words_sent]
    n_sent = length(tkns)
    for i in 1:n_sent
        low = max(i - 2, 1)
        high = min(i + 2, n_sent)
        m = tkns[i]
        if low < i
            for j in low:(i-1)
                if haskey(w_before[m], tkns[j])
                    w_before[m][tkns[j]] += 1.0f0
                else
                    w_before[m][tkns[j]] = 1.0f0
                end
            end
        end
        if i < high
            for j in (i+1):high
                if haskey(w_after[m], tkns[j])
                    w_after[m][tkns[j]] += 1.0f0
                else
                    w_after[m][tkns[j]] = 1.0f0
                end
            end
        end
    end
end

# ╔═╡ a65dbc6b-d5e8-4154-abb3-d50c4c68ee26
function mtx_from_word_ctx2(n_words::Int, w_before::Vector{Dict{Int,Float32}}, w_after::Vector{Dict{Int,Float32}}, xdims::Int)::Tuple{Matrix{Float32},Vector{Float32}}
    rows = Vector{Int}()
    cols = Vector{Int}()
    vals = Vector{Float32}()
    for i in 1:n_words
        if i % 10_000 == 1
            @info i
        end
        t = sum(values(w_before[i]))
        for (k, cnt) in w_before[i]
            push!(rows, k)
            push!(cols, i)
            push!(vals, cnt / t)
        end
        t = sum(values(w_after[i]))
        for (k, cnt) in w_after[i]
            push!(rows, n_words + k)
            push!(cols, i)
            push!(vals, cnt / t)
        end
    end
    wds_matrix = sparse(rows, cols, vals)
    @info wds_matrix
    u, s, v = rsvd(wds_matrix, xdims)
    wds_ps = convert(Matrix{Float32}, s .* v')
    wds_s = convert(Vector{Float32}, s)
    @info wds_ps, wds_s
    wds_ps, wds_s
end

# ╔═╡ 7ca79b00-4c3c-411e-be7f-4556c2e9205b
md"""
##### Build words vector from before and after words context
"""

# ╔═╡ 2d02ee1e-9247-4a60-8e2e-efec5498ae2a
md"""
**Prepare word context vectors:`  `**
$(@bind prepare_context CheckBox())
"""

# ╔═╡ a4f44f26-319b-48df-ad95-67d24a513a62
let
    if prepare_context
        time1 = now()
        w_before = [Dict{Int,Float32}() for i in 1:n_words]
        w_after = [Dict{Int,Float32}() for i in 1:n_words]
        for (i, sent) in enumerate(sents)
            if i % 10_000 == 1
                @info i
            end
            word_ctx2_add_sent(w_before, w_after, sent, word_idx)
        end
        @info w_before, w_after
        ps, s = mtx_from_word_ctx2(n_words, w_before, w_after, 200)
		xdims = 50
		word_ctx2_ps = ps[1:xdims, :]
        time2 = now()
        @info word_ctx2_ps
        serialize("../data/imdb_word_ctx2_vs.jls", word_ctx2_ps)
        @info "Used ", time_span(time1, time2, :Second)
        plot(1:length(s), s, label="Singular Values (word context)")
    end
end

# ╔═╡ 0088bc7d-768a-4e52-ae53-76e6f216a372
begin
    word_ctx2_ps = deserialize("../data/imdb_word_ctx2_vs.jls")
    @info word_ctx2_ps
end

# ╔═╡ 4f074f9e-1cdf-46ce-9755-352cfe8ef013
md"""
##### Partition the word contexts vectors
"""

# ╔═╡ df2d65ab-ac0c-4de9-bfef-c9e7dac8e6a1
md"""
**Clustering word context vectors:`  `**
$(@bind cluster_word_ctx CheckBox())
"""

# ╔═╡ 7dfc80f9-3ea6-4c73-bad7-e1b8836e7c08
let
    if cluster_word_ctx
        time1 = now()
        word_ctx2_cluster = vecs_cluster(word_ctx2_ps, 9, 100)
        time2 = now()
        serialize("../data/imdb_ctx2_cluster.jls", word_ctx2_cluster)
        @info "Used ", time_span(time1, time2, :Second)
    end
end

# ╔═╡ 5837df3a-393e-4824-bb28-8d7be45abb5c
begin
    word_ctx2_cluster = deserialize("../data/imdb_ctx2_cluster.jls")
    @info length(word_ctx2_cluster.cells), word_ctx2_cluster
end

# ╔═╡ 9aad60ce-6865-4b02-99e3-cf9e29c2e475
md"""
##### Display words by meaning
"""

# ╔═╡ ce44673b-92bf-4013-83db-d0e7852f4327
md"""
**Word Context Vectors Cell Index:`  `**
$(@bind cell_index_ctx NumberField(1:length(word_ctx2_cluster.cells), default=1))
"""

# ╔═╡ 03708a89-eded-4c9d-9b3e-e5cb92bcd97b
let
    ws = []
    for i in word_ctx2_cluster.cells[cell_index_ctx].indices
        push!(ws, words[i])
    end
    sort!(ws)
    for w in ws
        println(w)
    end
end

# ╔═╡ f8d392c2-645a-44ff-b7c9-8afa6fa7cdf0
md"""
**Word index by meaning:`  `**
$(@bind w_index_ctx NumberField(1:length(words), default=1))
"""

# ╔═╡ e254b1d6-6d16-446b-b80a-15e6c512b250
if isa(w_index_ctx, Int) && w_index_ctx >= 1 && w_index_ctx <= n_words
    print(words[w_index_ctx])
end

# ╔═╡ f66220dc-bb36-476f-a431-b3afa46c87b0
let
    if isa(w_index_ctx, Int) && w_index_ctx >= 1 && w_index_ctx <= n_words
        x = word_ctx2_ps[:, w_index_ctx]
        _, c = closest_cell(word_ctx2_cluster, x)
        ks, d2 = top_match_samples(word_ctx2_cluster, x, c, 10)
        for (i, k) in enumerate(ks)
            println(words[k], " - ", d2[i])
        end
    end
end

# ╔═╡ Cell order:
# ╠═76fbeff9-3a2b-48db-bd5e-fb5afce8e701
# ╠═e09d6b44-6370-11ed-1d3e-3fa29ee591a3
# ╠═7d65c9d5-6dd7-45ac-8aee-b0b1d94329b9
# ╠═0ead4006-0a2d-4eb5-bac9-23077729e79b
# ╠═3bb7c7a8-d8cf-4f39-ab31-f7d670ad3b49
# ╠═e7549436-2c9d-4edb-88aa-3e443c7be581
# ╠═430c58e1-b25c-4fa3-91e2-8b77b62518a7
# ╠═a65dbc6b-d5e8-4154-abb3-d50c4c68ee26
# ╟─7ca79b00-4c3c-411e-be7f-4556c2e9205b
# ╟─2d02ee1e-9247-4a60-8e2e-efec5498ae2a
# ╠═a4f44f26-319b-48df-ad95-67d24a513a62
# ╠═0088bc7d-768a-4e52-ae53-76e6f216a372
# ╟─4f074f9e-1cdf-46ce-9755-352cfe8ef013
# ╟─df2d65ab-ac0c-4de9-bfef-c9e7dac8e6a1
# ╠═7dfc80f9-3ea6-4c73-bad7-e1b8836e7c08
# ╠═5837df3a-393e-4824-bb28-8d7be45abb5c
# ╟─9aad60ce-6865-4b02-99e3-cf9e29c2e475
# ╟─ce44673b-92bf-4013-83db-d0e7852f4327
# ╟─03708a89-eded-4c9d-9b3e-e5cb92bcd97b
# ╟─f8d392c2-645a-44ff-b7c9-8afa6fa7cdf0
# ╟─e254b1d6-6d16-446b-b80a-15e6c512b250
# ╟─f66220dc-bb36-476f-a431-b3afa46c87b0
