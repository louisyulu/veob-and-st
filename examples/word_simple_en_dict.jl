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
    Serialization, JSON, HypertextLiteral

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

# ╔═╡ 8f6b33ca-04a2-42dd-8e7b-ef2dd90aae27
begin
    if !isdir("../data")
        mkdir("../data")
    end
    if !isdir("../text")
        mkdir("../text")
    end
end

# ╔═╡ 4f321908-9978-46bb-a95e-f12331d652ce
md"""
##### Word entries of merged.json file from https://github.com/nightblade9/simple-english-dictionary
"""

# ╔═╡ 4962414f-0007-4438-b7fd-b307a7bbfbc1
# the string are all in lower case
mutable struct WordEntry
    word::String
    synonyms::Vector{String}
    antonyms::Vector{String}
end

# ╔═╡ e7549436-2c9d-4edb-88aa-3e443c7be581
let
    fd = open("../text/simple-english-dictionary.json", "r")
    f_txt = replace(read(fd, String), !isascii => '~')
    close(fd)
    w_entries = JSON.parse(f_txt)
    global n_entries = length(w_entries)
    @info n_entries, w_entries
    global word_entries = Vector{WordEntry}(undef, n_entries)
    i = 1
    for (w, b) in w_entries
        word_entries[i] = WordEntry(
            lowercase(w),
            [lowercase(s) for s in b["SYNONYMS"]],
            [lowercase(a) for a in b["ANTONYMS"]]
        )
        push!(word_entries[i].synonyms, lowercase(w)) # inclue the word itself in synonyms
        i += 1
    end
    sort!(word_entries, by=e -> e.word)
    @info word_entries
end

# ╔═╡ 7676ed9f-dcf3-4a26-bff4-7eb5b14a854d
begin
    words = [e.word for e in word_entries]
    n_words = length(words)
    word_idx = Dict{String,Int}()
    for (i, w) in enumerate(words)
        word_idx[w] = i
    end
    @info n_words, words, word_idx

    local syn_set = Set{String}()
    local ant_set = Set{String}()
    for e in word_entries
        union!(syn_set, e.synonyms)
        union!(ant_set, e.antonyms)
    end

    synonyms = sort(collect(syn_set))
    n_synonyms = length(synonyms)
    synonym_idx = Dict{String,Int}()
    for (i, w) in enumerate(synonyms)
        synonym_idx[w] = i
    end
    @info n_synonyms, synonyms, synonym_idx

    antonyms = sort(collect(ant_set))
    n_antonyms = length(antonyms)
    antonym_idx = Dict{String,Int}()
    for (i, w) in enumerate(antonyms)
        antonym_idx[w] = i
    end
    @info n_antonyms, antonyms, antonym_idx
end

# ╔═╡ 1a100112-08d0-4583-a4a0-e302f0768fe2
let
    l_words = [length(w) for w in words]
    @info "Word statistics:", n_words, mean(l_words), std(l_words)
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
        N = 5
        m_max = 5
        n_chars = 127
        for (i, chs) in enumerate(w_ch_pos)
            for (ch, pos) in chs
                for m in 1:m_max
                    v = sum([cos_fn(N, m, k) for k in pos])
                    push!(cols, i)
                    push!(rows, n_chars * (m - 1) + ch)
                    push!(vals, v)
                end
            end
        end
        wds_matrix = collect(sparse(rows, cols, vals))
        @info wds_matrix
        xdims = 128
        u, s, v = rsvd(wds_matrix, xdims)
        w_cos_ps = convert(Matrix{Float32}, s .* v' / norm(s))
        w_cos_ut = convert(Matrix{Float32}, u')
        w_cos_s = convert(Vector{Float32}, s)
        time2 = now()
        @info w_cos_ps, w_cos_ut, w_cos_s
        serialize("../data/sed_word_cos_vs.jls", (w_cos_ps, w_cos_ut, w_cos_s))
        @info "Used ", time_span(time1, time2, :Second)
        plot(1:length(s), s, label="Singular Values (Cos Transformed)")
    end
end

# ╔═╡ 44107156-8803-448c-ac1c-c3d08a4a7685
begin
    w_cos_ps, w_cos_ut, w_cos_s = deserialize("../data/sed_word_cos_vs.jls")
    w_dims = size(w_cos_ps, 1)
    @info w_dims, w_cos_ps, w_cos_ut, w_cos_s
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
        word_cos_cluster = vecs_cluster(w_cos_ps, 10, 100)
        time2 = now()
        serialize("../data/sed_word_cos_cluster.jls", word_cos_cluster)
        @info "Used ", time_span(time1, time2, :Second)
    end
end

# ╔═╡ e44f84e2-b499-42e6-854c-28b8544964f4
begin
    word_cos_cluster = deserialize("../data/sed_word_cos_cluster.jls")
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

# ╔═╡ abee752d-45db-4b36-b6b1-605c69f6d41a
md"""
**Prepare word semantic vectors:`  `**
$(@bind prepare_semantic CheckBox())
"""

# ╔═╡ 7ef4a156-bfdf-415c-ac7b-e29b407c5380
let
    if prepare_semantic
        time1 = now()
        rows = Vector{Int}()
        cols = Vector{Int}()
        vals = Vector{Float32}()
        for (i, e) in enumerate(word_entries)
            if i % 10_000 == 1
                @info i
            end
            i_w = word_idx[e.word]
            for (j, v) in enumerate(w_cos_ps[:, i_w])
                push!(cols, i_w)
                push!(rows, j)
                push!(vals, v)
            end
            for s in Set(e.synonyms)
                i_s = synonym_idx[s]
                push!(cols, i_w)
                push!(rows, w_dims + i_s)
                push!(vals, 1.0f0)
            end
            for a in Set(e.antonyms)
                i_a = antonym_idx[a]
                push!(cols, i_w)
                push!(rows, w_dims + n_synonyms + i_a)
                push!(vals, 1.0f0)
            end
        end
        semantic_matrix = sparse(rows, cols, vals)
        @info semantic_matrix
        s_dims = 300
        u, s, v = rsvd(semantic_matrix, s_dims)
        semantic_ps = convert(Matrix{Float32}, s .* v')
        @info semantic_ps, s
        time2 = now()
        serialize("../data/sed_word_semantic_vs.jls", semantic_ps)
        @info "Used ", time_span(time1, time2, :Second)
        plot(1:length(s), s, label="Singular Values (word semantic)")
    end
end

# ╔═╡ 2f81cac2-fe41-4f90-a8bf-de7f9fda071b
begin
    semantic_ps = deserialize("../data/sed_word_semantic_vs.jls")
    @info semantic_ps
end

# ╔═╡ 75dd599b-3928-4176-9a0a-5270c3fab6e1
md"""
**Clustering word semantic vectors:`  `**
$(@bind cluster_semantic_vs CheckBox())
"""

# ╔═╡ a2b6a631-4806-4fdb-ac7d-1cdbd78c0185
let
    if cluster_semantic_vs
        time1 = now()
        semantic_cluster = vecs_cluster(semantic_ps, 10, 100)
        time2 = now()
        serialize("../data/sed_word_semantic_cluster.jls", semantic_cluster)
        @info "Used ", time_span(time1, time2, :Second)
    end
end

# ╔═╡ 6a5ef758-6ca9-43e0-80e8-d88104a2ee55
begin
    semantic_cluster = deserialize("../data/sed_word_semantic_cluster.jls")
    @info length(semantic_cluster.cells), semantic_cluster
end

# ╔═╡ 33ffe585-72f5-4a49-b0dc-d2e466f0e294
md"""
**Semantic Cell Index:`  `**
$(@bind cell_semantic_idx NumberField(1:length(semantic_cluster.cells), default=1))
"""

# ╔═╡ 0cab2487-9fc7-4ff0-bd0f-235fc1c02fd7
let
    ws = []
    for i in semantic_cluster.cells[cell_semantic_idx].indices
        push!(ws, words[i])
    end
    sort!(ws)
    for w in ws
        println(w)
    end
end

# ╔═╡ dfc700ac-42cd-4b76-a766-453257a8d4fc
md"""
**Word index by semantic:`  `**
$(@bind semantic_idx NumberField(1:length(words), default=1))
"""

# ╔═╡ d3e07e74-20f6-4651-a864-2a2e6014baf8
if isa(semantic_idx, Int) && semantic_idx >= 1 && semantic_idx <= n_words
    print(words[semantic_idx])
end

# ╔═╡ 7468cb6a-a014-47b6-93f2-3e937da91c6b
let
    if isa(semantic_idx, Int) && semantic_idx >= 1 && semantic_idx <= n_words
        x = semantic_ps[:, semantic_idx]
        _, c = closest_cell(semantic_cluster, x)
        ks, d2 = top_match_samples(semantic_cluster, x, c, 10)
        for (i, k) in enumerate(ks)
            println(words[k], " - ", d2[i])
        end
    end
end

# ╔═╡ 853577e8-7299-433b-b7de-9bcd2fd75a31
function token_spell_vec(tkn::String, ut::Matrix{Float32}, sv::Vector{Float32}, ch_max::Int=127, idx_lim::Int=5)::Vector{Float32}
    ch_pos = Dict{Int,Vector{Int}}()
    for (i, ch) in enumerate(tkn)
        i_ch = Int(ch)
        if !haskey(ch_pos, i_ch)
            ch_pos[i_ch] = [i]
        else
            push!(ch_pos[i_ch], i)
        end
    end
    N = 5
    m_max = 5
    rows = Int[]
    vals = Float32[]
    for (ch, pos) in ch_pos
        for m in 1:m_max
            v = sum([cos_fn(N, m, k) for k in pos])
            push!(rows, ch)
            push!(vals, v)
        end
    end
    sp_v = sparsevec(rows, vals, size(ut, 2))
    ut * collect(sp_v) / norm(sv)
end

# ╔═╡ 6c7d0284-4c53-44b1-b671-12669ef7118a
md"""
**Testing Word:`  `**
$(@bind testing_word TextField(default="text"))
"""

# ╔═╡ 40c73e97-a267-41fc-abae-a0108b18fa60
let
    x = token_spell_vec(lowercase(testing_word), w_cos_ut, w_cos_s)
    _, c = closest_cell(word_cos_cluster, x)
    ks, d2 = top_match_samples(word_cos_cluster, x, c, 7)
    global word_options = []
    global option_idx = Dict{String,Int}()
    for (i, k) in enumerate(ks)
        w = "$(words[k]) - $(d2[i])"
        push!(word_options, w)
        option_idx[w] = k
    end
end

# ╔═╡ c716deb9-3c28-49cf-b360-3158e01d8f8d
md"""
$(@bind closest_words MultiSelect(word_options))
"""

# ╔═╡ f64d72ad-3cf0-47f7-8e30-50e966ad5cb0
let
    if length(closest_words) > 0
        wd = closest_words[1]
        idx = option_idx[wd]
        x = semantic_ps[:, idx]
        _, c = closest_cell(semantic_cluster, x)
        ks, d2 = top_match_samples(semantic_cluster, x, c, 10)
        for (i, k) in enumerate(ks)
            println(words[k], " ", d2[i])
        end
    end
end

# ╔═╡ Cell order:
# ╠═76fbeff9-3a2b-48db-bd5e-fb5afce8e701
# ╠═e09d6b44-6370-11ed-1d3e-3fa29ee591a3
# ╠═7d65c9d5-6dd7-45ac-8aee-b0b1d94329b9
# ╠═0ead4006-0a2d-4eb5-bac9-23077729e79b
# ╠═3bb7c7a8-d8cf-4f39-ab31-f7d670ad3b49
# ╠═8f6b33ca-04a2-42dd-8e7b-ef2dd90aae27
# ╟─4f321908-9978-46bb-a95e-f12331d652ce
# ╠═4962414f-0007-4438-b7fd-b307a7bbfbc1
# ╠═e7549436-2c9d-4edb-88aa-3e443c7be581
# ╠═7676ed9f-dcf3-4a26-bff4-7eb5b14a854d
# ╠═1a100112-08d0-4583-a4a0-e302f0768fe2
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
# ╟─abee752d-45db-4b36-b6b1-605c69f6d41a
# ╠═7ef4a156-bfdf-415c-ac7b-e29b407c5380
# ╠═2f81cac2-fe41-4f90-a8bf-de7f9fda071b
# ╟─75dd599b-3928-4176-9a0a-5270c3fab6e1
# ╠═a2b6a631-4806-4fdb-ac7d-1cdbd78c0185
# ╠═6a5ef758-6ca9-43e0-80e8-d88104a2ee55
# ╟─33ffe585-72f5-4a49-b0dc-d2e466f0e294
# ╟─0cab2487-9fc7-4ff0-bd0f-235fc1c02fd7
# ╟─dfc700ac-42cd-4b76-a766-453257a8d4fc
# ╟─d3e07e74-20f6-4651-a864-2a2e6014baf8
# ╟─7468cb6a-a014-47b6-93f2-3e937da91c6b
# ╠═853577e8-7299-433b-b7de-9bcd2fd75a31
# ╟─6c7d0284-4c53-44b1-b671-12669ef7118a
# ╠═40c73e97-a267-41fc-abae-a0108b18fa60
# ╟─c716deb9-3c28-49cf-b360-3158e01d8f8d
# ╟─f64d72ad-3cf0-47f7-8e30-50e966ad5cb0
