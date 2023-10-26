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
    Serialization, HypertextLiteral, PythonCall

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

# ╔═╡ 4f321908-9978-46bb-a95e-f12331d652ce
md"""
##### Load IMDB movie review dataset and extract letters only words and sentences
"""

# ╔═╡ e7549436-2c9d-4edb-88aa-3e443c7be581
begin
    words, word_idx, sents, sent_src = deserialize("../data/imdb_words_sents.jls")
    n_words = length(words)
    @info n_words, words, word_idx, length(sents), sents, sent_src
end

# ╔═╡ 50f2f4d3-e4db-43f7-94f1-31bd8f6eb472
md"""
**Prepare word vectors (spelling):`  `**
$(@bind prepare_word_vecs CheckBox())
"""

# ╔═╡ 9c03b907-8ad1-4577-95ed-f90e347e1a7e
let
    if prepare_word_vecs
        time1 = now()
        w_ch_pos = [Dict{Int,Vector{Int}}() for i in 1:n_words]
        for (i, wd) in enumerate(words)
            for (j, ch) in enumerate(wd)
                i_ch = Int(ch)
                if !haskey(w_ch_pos[i], i_ch)
                    w_ch_pos[i][i_ch] = [j]
                else
                    push!(w_ch_pos[i][i_ch], j)
                end
            end
        end
        @info w_ch_pos
        cols = Int[]
        rows = Int[]
        vals = Float32[]
        N = 15
        m_max = 6
        for (i, chs) in enumerate(w_ch_pos)
            for (ch, pos) in chs
                for m in 1:m_max
                    v = sum([cos_fn(N, m, k) for k in pos])
                    push!(cols, i)
                    push!(rows, ch)
                    push!(vals, v)
                end
            end
        end
        wds_matrix = collect(sparse(rows, cols, vals))
        @info wds_matrix
        xdims = 40
        u, s, v = svd(wds_matrix)
        w_cos_ps = convert(Matrix{Float32}, s .* v' / s[1])[1:xdims, :]
        w_cos_ut = convert(Matrix{Float32}, u')[1:xdims, :]
        time2 = now()
        @info w_cos_ps, w_cos_ut
        serialize("../data/imdb_word_cos_vs.jls", (w_cos_ps, w_cos_ut))
        @info "Used ", time_span(time1, time2, :Second)
        plot(1:length(s), s, label="Singular Values (Cos Transformed)")
    end
end

# ╔═╡ 44107156-8803-448c-ac1c-c3d08a4a7685
begin
    w_cos_ps, w_cos_ut = deserialize("../data/imdb_word_cos_vs.jls")
    @info w_cos_ps, w_cos_ut
end

# ╔═╡ d0c85ca7-4861-488f-a0f4-8e8ddfe9ad4d
md"""
**Clustering word vectors (spelling):`  `**
$(@bind cluster_word_vecs CheckBox())
"""

# ╔═╡ d0e11884-4933-4974-94b4-222ee9917d98
let
    if cluster_word_vecs
        time1 = now()
        word_cos_cluster = vecs_cluster(w_cos_ps, 9, 100)
        time2 = now()
        serialize("../data/imdb_word_cos_cluster.jls", word_cos_cluster)
        @info "Used ", time_span(time1, time2, :Second)
    end
end

# ╔═╡ e44f84e2-b499-42e6-854c-28b8544964f4
begin
    word_cos_cluster = deserialize("../data/imdb_word_cos_cluster.jls")
    @info length(word_cos_cluster.cells), word_cos_cluster
end

# ╔═╡ cc1f619f-c86e-47d0-bab1-9b288e689c06
md"""
**Spelling Cell Index:`  `**
$(@bind cell_index NumberField(1:length(word_cos_cluster.cells), default=1))
"""

# ╔═╡ 70af4c59-b6f0-4d8c-9440-2dae403ea3ba
let
    ws = []
    for i in word_cos_cluster.cells[cell_index].indices
        push!(ws, words[i])
    end
    sort!(ws)
    for w in ws
        println(w)
    end
end

# ╔═╡ 8ccfd53c-6857-4cb3-a7d0-07e24293c570
md"""
**Word index by spelling:`  `**
$(@bind w_index NumberField(1:length(words), default=1))
"""

# ╔═╡ 8f3835ae-2a24-4dfe-85c3-c0e6737b3135
if isa(w_index, Int) && w_index >= 1 && w_index <= n_words
    print(words[w_index])
end

# ╔═╡ a55f98a2-9067-4d33-bdf7-cfd680eb7500
let
    if isa(w_index, Int) && w_index >= 1 && w_index <= n_words
        x = w_cos_ps[:, w_index]
        _, c = closest_cell(word_cos_cluster, x)
        ks, d2 = top_match_samples(word_cos_cluster, x, c, 10)
        for (i, k) in enumerate(ks)
            println(words[k], " - ", d2[i])
        end
    end
end

# ╔═╡ 2f5e4e31-b4d0-4998-a5f5-5259d37cd90a
let
    x = [cos_fn(16, 1, k) for k in 1:16]
    plot(1:length(x), x, label="cos transform")
end

# ╔═╡ Cell order:
# ╠═76fbeff9-3a2b-48db-bd5e-fb5afce8e701
# ╠═e09d6b44-6370-11ed-1d3e-3fa29ee591a3
# ╠═7d65c9d5-6dd7-45ac-8aee-b0b1d94329b9
# ╠═0ead4006-0a2d-4eb5-bac9-23077729e79b
# ╠═3bb7c7a8-d8cf-4f39-ab31-f7d670ad3b49
# ╟─4f321908-9978-46bb-a95e-f12331d652ce
# ╠═e7549436-2c9d-4edb-88aa-3e443c7be581
# ╟─50f2f4d3-e4db-43f7-94f1-31bd8f6eb472
# ╠═9c03b907-8ad1-4577-95ed-f90e347e1a7e
# ╠═44107156-8803-448c-ac1c-c3d08a4a7685
# ╟─d0c85ca7-4861-488f-a0f4-8e8ddfe9ad4d
# ╠═d0e11884-4933-4974-94b4-222ee9917d98
# ╠═e44f84e2-b499-42e6-854c-28b8544964f4
# ╟─cc1f619f-c86e-47d0-bab1-9b288e689c06
# ╟─70af4c59-b6f0-4d8c-9440-2dae403ea3ba
# ╟─8ccfd53c-6857-4cb3-a7d0-07e24293c570
# ╟─8f3835ae-2a24-4dfe-85c3-c0e6737b3135
# ╟─a55f98a2-9067-4d33-bdf7-cfd680eb7500
# ╠═2f5e4e31-b4d0-4998-a5f5-5259d37cd90a
