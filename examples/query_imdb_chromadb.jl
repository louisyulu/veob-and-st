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
using PlutoUI, Plots, Logging, Dates, LinearAlgebra, Serialization, HypertextLiteral, PythonCall

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

# ╔═╡ 010f28a3-5e3a-43ac-9a74-ee382434d44f
if !isdir("../data/chromadb")
    mkdir("../data/chromadb")
end

# ╔═╡ 62a1615d-cc8a-436d-8663-bf299ecf37a3
md"""
##### Query IMDB words and sentences saved in Chroma DB
"""

# ╔═╡ d59d4f73-170f-4a58-8af3-e5dbe715782d
begin
    words, word_idx, sents, sent_src = deserialize("../data/imdb_words_sents.jls")
    n_words = length(words)
    n_sents = length(sents)
    @info "Words:", n_words, words, word_idx, n_sents, sent_src

    word_spell_ps, word_spell_ut = deserialize("../data/imdb_word_cos_vs.jls")
    @info "Word vectors (spelling):", word_spell_ps, word_spell_ut

    word_ctx2_ps = deserialize("../data/imdb_word_ctx2_vs.jls")
    @info "Word vectors (graph):", word_ctx2_ps

    sent_svd_ps, sent_svd_ut = deserialize("../data/imdb_sent_cos_svd_vs.jls")
    @info "Sentence vectors (svd):", sent_svd_ps, sent_svd_ut

    sentence_vs = deserialize("../data/imdb_sent_cos_vs.jls")
    @info "Sentence vectors:", sentence_vs
end

# ╔═╡ 06e3c118-98a3-472e-9067-0e5ad2c730c7
md"""
##### Search word with similar spelling
"""

# ╔═╡ 1fb5608f-b8bd-499b-a240-8700a7864444
begin
    chromadb = pyimport("chromadb")
    client = chromadb.PersistentClient(path="../data/chromadb")
end

# ╔═╡ 69918ebc-5697-430b-97c6-5495ea16bbe4
begin
    coll_spelling = client.get_or_create_collection("word_spelling")
    coll_word_ctx2 = client.get_or_create_collection("word_ctx2")
    coll_sent_cos = client.get_or_create_collection("sentence_cos")
    coll_sent_vecs = client.get_or_create_collection("sentence_vecs")
end

# ╔═╡ 547f1216-0a90-4329-92f5-57737514214d
begin
    # client.delete_collection(name="word_spelling")
    # client.delete_collection(name="word_ctx2")
    # client.delete_collection(name="sentence_cos")
    # client.delete_collection(name="sentence_vecs")
end

# ╔═╡ b7c8db97-43a6-4ec9-9944-8f00ba65f1ac
md"""
**Load word (spelling and meaning) vectors:`  `**
$(@bind load_word_vs CheckBox())
"""

# ╔═╡ bb2f4119-77f8-4d9c-8418-be8253a36b33
let
    if load_word_vs
        time1 = now()
        i = 1
        n_batch = 1000
        while i < n_words
            i2 = if i + n_batch - 1 < n_words
                i + n_batch - 1
            else
                n_words
            end
            @info i, i2
            coll_spelling.upsert(
                documents=pylist(words[i:i2]),
                embeddings=pycollist(word_spell_ps[:, i:i2]),
                ids=pylist(["$k" for k in i:i2])
            )
            coll_word_ctx2.upsert(
                documents=pylist(words[i:i2]),
                embeddings=pycollist(word_ctx2_ps[:, i:i2]),
                ids=pylist(["$k" for k in i:i2])
            )
            i = i2 + 1
        end
        time2 = now()
        @info "Used ", time_span(time1, time2, :Minute)
        @info coll_spelling.peek(1), coll_word_ctx2.peek(1)
    end
end

# ╔═╡ a28d5399-9391-4a38-8b52-33fba02170a7
md"""
**Load sentence vectors:`  `**
$(@bind load_sent_vs CheckBox())
"""

# ╔═╡ 71958cc2-a537-4956-917b-c9f8a95071b2
let
    if load_sent_vs
        time1 = now()
        i = 1
        n_batch = 1000
        while i < n_sents
            i2 = if i + n_batch - 1 < n_sents
                i + n_batch - 1
            else
                n_sents
            end
            @info i, i2
            s_sents = [join(sents[j], ' ') for j in i:i2]
            meta = [pydict(["src" => sent_src[j]]) for j in i:i2]
            coll_sent_cos.upsert(
                documents=pylist(s_sents),
                embeddings=pycollist(sent_svd_ps[:, i:i2]),
                ids=pylist(["$k" for k in i:i2]),
                metadatas=pylist(meta)
            )
            coll_sent_vecs.upsert(
                documents=pylist(s_sents),
                embeddings=pycollist(sentence_vs[:, i:i2]),
                ids=pylist(["$k" for k in i:i2]),
                metadatas=pylist(meta)
            )
            i = i2 + 1
        end
        time2 = now()
        @info "Used ", time_span(time1, time2, :Minute)
        @info coll_sent_cos.peek(1)
    end
end

# ╔═╡ 4b6ed7dd-9c0c-4787-9f18-9a5578603a56
md"""
##### Search words with similar spelling
"""

# ╔═╡ bdbda227-de24-4887-9cee-fbc277e5f34d
md"""
**Word (Spelling) Index:`  `**
$(@bind word_sp_idx NumberField(1:n_words, default=1))
"""

# ╔═╡ ccb1e1b9-b981-4de6-aaf7-cef61372d197
let
    println(words[word_sp_idx])
end

# ╔═╡ 18350b75-19db-4a1a-9978-5eff938bbc29
let
    top_k = 10
    q_rs = coll_spelling.query(
        query_embeddings=pycollist(word_spell_ps[:, word_sp_idx]),
        n_results=top_k
    )
    rs = pyconvert(Dict, q_rs)
    wds = pyconvert(Vector, rs["documents"][1])
    ds = pyconvert(Vector, rs["distances"][1])
    for (i, w) in enumerate(wds)
        println(w, " - ", ds[i])
    end
end

# ╔═╡ e463adf6-d63d-48b0-aed8-41d80899afd1
md"""
##### Search words with related meaning
"""

# ╔═╡ 2df2ec06-1e18-4cb3-b4de-daa756600482
md"""
**Word (Graph) Index:`  `**
$(@bind word_graph_idx NumberField(1:n_words, default=1))
"""

# ╔═╡ e5a572dd-1b58-47f0-bb32-400f437c22be
let
    println(words[word_graph_idx])
end

# ╔═╡ 2b0f90f6-5c85-491f-bfb9-7f3dbc0d90d0
let
    top_k = 10
    q_rs = coll_word_ctx2.query(
        query_embeddings=pycollist(word_ctx2_ps[:, word_graph_idx]),
        n_results=top_k
    )
    rs = pyconvert(Dict, q_rs)
    wds = pyconvert(Vector, rs["documents"][1])
    ds = pyconvert(Vector, rs["distances"][1])
    for (i, w) in enumerate(wds)
        println(w, " - ", ds[i])
    end
end

# ╔═╡ c33dfa5c-37f6-4a6a-87bf-732d7125dd94
md"""
##### Search sentences
"""

# ╔═╡ fbf6d057-aa3c-48eb-b125-f955499da7b2
md"""
**Sentence (cos) Index:`  `**
$(@bind sent_cos_idx NumberField(1:n_sents, default=1))
"""

# ╔═╡ 25a83dd9-a7a9-415d-a5d0-b2ecda451838
let
    sent = sents[sent_cos_idx]
    println(join(sent, ' '), " - ", sent_src[sent_cos_idx])
end

# ╔═╡ 8df5865b-0bf6-4ea9-9882-9b3f3dc312b1
md"""
###### Sentence matches with svd compressed
"""

# ╔═╡ 1fbf8d56-4178-499f-9fde-5a2cecac31ab
let
    top_k = 7
    q_rs = coll_sent_cos.query(
        query_embeddings=pycollist(sent_svd_ps[:, sent_cos_idx]),
        n_results=top_k
    )
    rs = pyconvert(Dict, q_rs)
    ss = pyconvert(Vector, rs["documents"][1])
    ds = pyconvert(Vector, rs["distances"][1])
    ms = pyconvert(Vector, rs["metadatas"][1])
    for (i, s) in enumerate(ss)
        println(">> ", s, " - ", ds[i], " - ", ms[i]["src"], "\n")
    end
end

# ╔═╡ 2e00ed05-ae42-4b6e-9f09-349bf4e328b2
md"""
###### Sentence vectors matches
"""

# ╔═╡ f9903b36-f2b4-43fe-9763-9dcab44acce9
let
    top_k = 7
    q_rs = coll_sent_vecs.query(
        query_embeddings=pycollist(sentence_vs[:, sent_cos_idx]),
        n_results=top_k
    )
    rs = pyconvert(Dict, q_rs)
    ss = pyconvert(Vector, rs["documents"][1])
    ds = pyconvert(Vector, rs["distances"][1])
    ms = pyconvert(Vector, rs["metadatas"][1])
    for (i, s) in enumerate(ss)
        println(">> ", s, " - ", ds[i], " - ", ms[i]["src"], "\n")
    end
end

# ╔═╡ Cell order:
# ╠═76fbeff9-3a2b-48db-bd5e-fb5afce8e701
# ╠═e09d6b44-6370-11ed-1d3e-3fa29ee591a3
# ╠═7d65c9d5-6dd7-45ac-8aee-b0b1d94329b9
# ╠═0ead4006-0a2d-4eb5-bac9-23077729e79b
# ╠═3bb7c7a8-d8cf-4f39-ab31-f7d670ad3b49
# ╠═010f28a3-5e3a-43ac-9a74-ee382434d44f
# ╟─62a1615d-cc8a-436d-8663-bf299ecf37a3
# ╠═d59d4f73-170f-4a58-8af3-e5dbe715782d
# ╟─06e3c118-98a3-472e-9067-0e5ad2c730c7
# ╠═1fb5608f-b8bd-499b-a240-8700a7864444
# ╠═69918ebc-5697-430b-97c6-5495ea16bbe4
# ╠═547f1216-0a90-4329-92f5-57737514214d
# ╟─b7c8db97-43a6-4ec9-9944-8f00ba65f1ac
# ╠═bb2f4119-77f8-4d9c-8418-be8253a36b33
# ╟─a28d5399-9391-4a38-8b52-33fba02170a7
# ╠═71958cc2-a537-4956-917b-c9f8a95071b2
# ╟─4b6ed7dd-9c0c-4787-9f18-9a5578603a56
# ╟─bdbda227-de24-4887-9cee-fbc277e5f34d
# ╟─ccb1e1b9-b981-4de6-aaf7-cef61372d197
# ╟─18350b75-19db-4a1a-9978-5eff938bbc29
# ╟─e463adf6-d63d-48b0-aed8-41d80899afd1
# ╟─2df2ec06-1e18-4cb3-b4de-daa756600482
# ╟─e5a572dd-1b58-47f0-bb32-400f437c22be
# ╟─2b0f90f6-5c85-491f-bfb9-7f3dbc0d90d0
# ╟─c33dfa5c-37f6-4a6a-87bf-732d7125dd94
# ╟─fbf6d057-aa3c-48eb-b125-f955499da7b2
# ╟─25a83dd9-a7a9-415d-a5d0-b2ecda451838
# ╟─8df5865b-0bf6-4ea9-9882-9b3f3dc312b1
# ╟─1fbf8d56-4178-499f-9fde-5a2cecac31ab
# ╟─2e00ed05-ae42-4b6e-9f09-349bf4e328b2
# ╟─f9903b36-f2b4-43fe-9763-9dcab44acce9
