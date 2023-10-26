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
using PlutoUI, Plots, Logging, Dates, LinearAlgebra, RandomizedLinAlg, Statistics, SparseArrays, Serialization, HypertextLiteral

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

# ╔═╡ 62a1615d-cc8a-436d-8663-bf299ecf37a3
md"""
##### Load IMDB movie review dataset prepared words, sentences and vector embeddings
"""

# ╔═╡ d59d4f73-170f-4a58-8af3-e5dbe715782d
begin
    words, word_idx, sents, sent_src = deserialize("../data/imdb_words_sents.jls")
    n_words = length(words)
    n_sents = length(sents)
    @info n_words, words, word_idx, n_sents, sents, sent_src
    # word_ps, word_ut = deserialize("../data/imdb_word_graph_vs.jls")
    word_ps = deserialize("../data/imdb_word_ctx2_vs.jls")
    wd_dims = size(word_ps, 1)
    @info wd_dims, word_ps
end

# ╔═╡ 8c04c883-4ba2-4cb1-b01c-574a3df8ecd1
md"""
##### Sentences similarity grouping
"""


# ╔═╡ 462be09c-d30f-477b-b195-8dd73f3a22fd
md"""
**Build sentences vectors:`  `**
$(@bind build_sent_vecs CheckBox())
"""

# ╔═╡ b0525a10-4aef-4555-bc81-8b20f06b0be3
let
    if build_sent_vecs
        time1 = now()
        s_w_pos = [Dict{Int,Vector{Int}}() for i in 1:n_sents]
        for (i, sent) in enumerate(sents)
            for (j, w) in enumerate(sent)
                i_w = word_idx[w]
                if !haskey(s_w_pos[i], i_w)
                    s_w_pos[i][i_w] = [j]
                else
                    push!(s_w_pos[i][i_w], j)
                end
            end
        end
        @info s_w_pos
        sentence_vs = zeros(Float32, 6 * wd_dims, n_sents)
        for (i, iw_pos) in enumerate(s_w_pos)
            if i % 10_000 == 1
                @info i
            end
            N = 32
            m_max = 6
            for (iw, ks) in iw_pos
                for m in 1:m_max
                    v = sum([word_ps[:, iw] * cos_fn(N, m, k) for k in ks])
                    sentence_vs[((m-1)*wd_dims+1):m*wd_dims, i] = v
                end
            end
        end
        @info sentence_vs
        st_dims = 200
        u, s, v = rsvd(sentence_vs, st_dims)
        sentence_ps = convert(Matrix{Float32}, s .* v')
        sentence_ut = convert(Matrix{Float32}, u')
        time2 = now()
        @info sentence_ps, sentence_ut
        serialize("../data/imdb_sent_cos_svd_vs.jls", (sentence_ps, sentence_ut))
        serialize("../data/imdb_sent_cos_vs.jls", sentence_vs)
        @info "Used ", time_span(time1, time2, :Minute)
        plot(1:length(s), s, label="Singular Values (Sentence)")
    end
end

# ╔═╡ 5f2e84a4-9bf9-44c5-8efe-6f5e106d416e
begin
    sentence_ps, sentence_ut = deserialize("../data/imdb_sent_cos_svd_vs.jls")
    st_dims = size(sentence_ps, 1)
    @info sentence_ps, sentence_ut
end

# ╔═╡ 4036e725-0084-4fbc-a17c-4d1270f34658
md"""
**Clustering sentences vectors:`  `**
$(@bind cluster_sent_vecs CheckBox())
"""

# ╔═╡ ade50c15-1c01-433b-9bb2-293453361e03
let
    if cluster_sent_vecs
        time1 = now()
        sentences_cluster = vecs_cluster(sentence_ps, 9, 100)
        time2 = now()
        serialize("../data/imdb_sent_cos_cluster.jls", sentences_cluster)
        @info "Used ", time_span(time1, time2, :Minute)
    end
end

# ╔═╡ 51865f1a-eb2e-4604-bdd6-002de99bfb80
begin
    sentences_cluster = deserialize("../data/imdb_sent_cos_cluster.jls")
    @info length(sentences_cluster.cells), sentences_cluster
end

# ╔═╡ 307d2b79-2293-4ff3-b61e-30dbe8f65b61
md"""
##### Display sentences cluster
"""

# ╔═╡ 93901247-b6a2-45da-994f-5c42f763809f
md"""
**Sentences Cell Index:`  `**
$(@bind sent_cell_idx NumberField(1:length(sentences_cluster.cells), default=1))
"""

# ╔═╡ 54fdabb1-c4dc-4f7d-99c5-b1e1a109244e
let
    # display first 100 sentences
    for (i, k) in enumerate(sentences_cluster.cells[sent_cell_idx].indices)
        sent = sents[k]
        println(">> ", join(sent, ' '))
        if i > 100
            break
        end
    end
end

# ╔═╡ 8b114313-8a0a-4274-9a2a-b58ab7e1add0
md"""
**Sentence Index:`  `**
$(@bind sent_idx NumberField(1:n_sents, default=1))
"""

# ╔═╡ 720bfc4d-d2da-46f9-a9d5-1e75eca1deff
println(join(sents[sent_idx], ' '), " - ", sent_src[sent_idx])

# ╔═╡ d58d6f1b-71e6-4689-8086-5a702d2fe540
let
    x = sentence_ps[:, sent_idx]
    _, c = closest_cell(sentences_cluster, x)
    ks, d2 = top_match_samples(sentences_cluster, x, c, 7)
    for (i, k) in enumerate(ks)
        println(">> ", join(sents[k], ' '), " ", d2[i], " - ", sent_src[k], "\n")
    end
end

# ╔═╡ Cell order:
# ╠═76fbeff9-3a2b-48db-bd5e-fb5afce8e701
# ╠═e09d6b44-6370-11ed-1d3e-3fa29ee591a3
# ╠═7d65c9d5-6dd7-45ac-8aee-b0b1d94329b9
# ╠═0ead4006-0a2d-4eb5-bac9-23077729e79b
# ╠═3bb7c7a8-d8cf-4f39-ab31-f7d670ad3b49
# ╟─62a1615d-cc8a-436d-8663-bf299ecf37a3
# ╠═d59d4f73-170f-4a58-8af3-e5dbe715782d
# ╟─8c04c883-4ba2-4cb1-b01c-574a3df8ecd1
# ╟─462be09c-d30f-477b-b195-8dd73f3a22fd
# ╠═b0525a10-4aef-4555-bc81-8b20f06b0be3
# ╠═5f2e84a4-9bf9-44c5-8efe-6f5e106d416e
# ╟─4036e725-0084-4fbc-a17c-4d1270f34658
# ╠═ade50c15-1c01-433b-9bb2-293453361e03
# ╠═51865f1a-eb2e-4604-bdd6-002de99bfb80
# ╟─307d2b79-2293-4ff3-b61e-30dbe8f65b61
# ╟─93901247-b6a2-45da-994f-5c42f763809f
# ╟─54fdabb1-c4dc-4f7d-99c5-b1e1a109244e
# ╟─8b114313-8a0a-4274-9a2a-b58ab7e1add0
# ╟─720bfc4d-d2da-46f9-a9d5-1e75eca1deff
# ╟─d58d6f1b-71e6-4689-8086-5a702d2fe540
