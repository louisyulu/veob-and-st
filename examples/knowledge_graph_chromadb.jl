### A Pluto.jl notebook ###
# v0.19.32
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
    Serialization, HypertextLiteral, CSV, DataFrames, PythonCall

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

# ╔═╡ e9d3aea1-b98f-4da3-aa17-ba2473959fe6
begin
    if !isdir("../text")
        mkdir("../text")
    end
    if !isdir("../data")
        mkdir("../data")
    end
    if !isdir("../data/knowledge_graph")
        mkdir("../data/knowledge_graph")
    end
end

# ╔═╡ 607395ef-b0d2-41c5-8159-32d2c91763f2
md"""
**Read in graph triples data:`  `**
$(@bind read_in_data CheckBox())
"""

# ╔═╡ e7549436-2c9d-4edb-88aa-3e443c7be581
let
    if read_in_data
        df = CSV.read("../text/conceptEdges.csv", DataFrame; header=[:head, :link, :tail])
        triples = Tuple{String,String,String}[(t[1], t[2], t[3]) for t in eachrow(df)]
        n_triples = length(triples)
        @info n_triples, triples

        head_set = Set{String}(df[:, :head])
        heads = collect(head_set)
        n_heads = length(heads)
        head_idx = Dict{String,Int}()
        for (i, h) in enumerate(heads)
            head_idx[h] = i
        end
        @info "head", n_heads, heads, head_idx

        link_set = Set{String}(df[:, :link])
        links = collect(link_set)
        n_links = length(links)
        link_idx = Dict{String,Int}()
        for (i, k) in enumerate(links)
            link_idx[k] = i
        end
        @info "link", n_links, links, link_idx

        tail_set = Set{String}(df[:, :tail])
        tails = collect(tail_set)
        n_tails = length(tails)
        tail_idx = Dict{String,Int}()
        for (i, t) in enumerate(tails)
            tail_idx[t] = i
        end
        @info "tail", n_tails, tails, tail_idx
        serialize("../data/knowledge-graph-data.jls", (triples, heads, head_idx, links, link_idx, tails, tail_idx))
    end
end

# ╔═╡ b8155ffa-8479-424e-83e7-b0c29e938fb6
begin
    triples, heads, head_idx, links, link_idx, tails, tail_idx = deserialize("../data/knowledge-graph-data.jls")
    n_triples = length(triples)
    @info n_triples, triples
    n_heads = length(heads)
    @info "head", n_heads, heads, head_idx
    n_links = length(links)
    @info "link", n_links, links, link_idx
    n_tails = length(tails)
    @info "tail", n_tails, tails, tail_idx
end

# ╔═╡ 1206c585-fa44-4f1b-9af2-fa12508c82f8
function spell_vecs(tokens::Vector{String})
    n_tokens = length(tokens)
    tkn_ch_pos = [Dict{Int,Vector{Int}}() for i in 1:n_tokens]
    for (i, tkn) in enumerate(tokens)
        for (j, ch) in enumerate(lowercase(tkn))
            i_ch = Int(ch)
            if !haskey(tkn_ch_pos[i], i_ch)
                tkn_ch_pos[i][i_ch] = [j]
            else
                push!(tkn_ch_pos[i][i_ch], j)
            end
        end
    end
    # @info tkn_ch_pos
    cols = Int[]
    rows = Int[]
    vals = Float32[]
    N = 5
    m_max = 5
    n_chars = 127
    for (i, chs) in enumerate(tkn_ch_pos)
        for (ch, pos) in sort(chs)
            if ch > n_chars
                ch = 32  # replace with space if non ascii
            end
            for m in 1:m_max
                v = sum([cos_fn(N, m, k) for k in pos])
                push!(cols, i)
                push!(rows, n_chars * (m - 1) + ch)
                push!(vals, v)
            end
        end
    end
    mtx = collect(sparse(rows, cols, vals, m_max * n_chars, n_tokens))
    # @info mtx
    xdims = 60
    u, s, v = rsvd(mtx, xdims)
    tkn_spell_ps = convert(Matrix{Float32}, s .* v')
    tkn_spell_ut = convert(Matrix{Float32}, u')
    tkn_spell_ps, tkn_spell_ut, s
end

# ╔═╡ 438c6e61-1ef2-40f2-b2cf-1ab907ea162d
md"""
**Build spelling vectors:`  `**
$(@bind build_spell_vs CheckBox())
"""

# ╔═╡ 16c0a8af-5580-4269-9b7b-87d87c14325e
let
    if build_spell_vs
        h_spell_ps, h_spell_ut, h_spell_s = spell_vecs(heads)
        @info h_spell_ps, h_spell_ut, h_spell_s
        t_spell_ps, t_spell_ut, t_spell_s = spell_vecs(tails)
        serialize("../data/knowledge-graph-spelling.jls", (h_spell_ps, h_spell_ut, h_spell_s, t_spell_ps, t_spell_ut, t_spell_s))
        @info t_spell_ps, t_spell_ut, t_spell_s
		plot(1:length(h_spell_s), h_spell_s, label="Singular Values (Spelling)")
    end
end

# ╔═╡ a8e9e421-7f67-42bd-bfc9-d6dd4fd2ff36
begin
    h_spell_ps, h_spell_ut, h_spell_s, t_spell_ps, t_spell_ut, t_spell_s = deserialize("../data/knowledge-graph-spelling.jls")
    @info h_spell_ps, h_spell_ut, h_spell_s, t_spell_ps, t_spell_ut, t_spell_s
end

# ╔═╡ d7d6377f-e3d9-43b3-974d-fd43acd07462
md"""
**Calculate vectors embedding:`  `**
$(@bind calc_embedding CheckBox())
"""

# ╔═╡ 9a2bb946-c658-49db-a6be-c1a056e4821d
let
    if calc_embedding
        rows = Vector{Int}()
        cols = Vector{Int}()
        vals = Vector{Float32}()
        for i in 1:n_triples
            t = triples[i]
            push!(rows, head_idx[t[1]])
            push!(cols, i)
            push!(vals, 1.0f0)

            push!(rows, n_heads + link_idx[t[2]])
            push!(cols, i)
            push!(vals, 1.0f0)

            push!(rows, n_heads + n_links + tail_idx[t[3]])
            push!(cols, i)
            push!(vals, 1.0f0)
        end
        n_rows = n_heads + n_links + n_tails
        mtx = sparse(rows, cols, vals, n_rows, n_triples)
        @info n_rows, n_triples, mtx
        xdims = 50
        u, s, v = rsvd(mtx, xdims)
        xs = s .* u'
        head_ps = convert(Matrix{Float32}, xs[:, 1:n_heads])
        link_ps = convert(Matrix{Float32}, xs[:, n_heads+1:n_heads+n_links])
        tail_ps = convert(Matrix{Float32}, xs[:, n_heads+n_links+1:end])
        @info head_ps, link_ps, tail_ps, u, s, v
        serialize("../data/knowledge-graph-vs.jls", (head_ps, link_ps, tail_ps))
        plot(1:length(s), s, label="Singular Values (Knowledge Graph)")
    end
end

# ╔═╡ 52c3206a-a670-401a-9f3f-df4367f64615
begin
    head_ps, link_ps, tail_ps = deserialize("../data/knowledge-graph-vs.jls")
    xdims = size(head_ps, 1)
    @info xdims, head_ps, link_ps, tail_ps
end

# ╔═╡ a3e05bcc-51e3-4c86-8b08-769dae5962fe
let
    global triple_ps = Matrix{Float32}(undef, 3 * xdims, n_triples)
    h_3_dict = Dict{Int,Vector{Int}}()
    t_3_dict = Dict{Int,Vector{Int}}()
    h_t_3_dict = Dict{String,Vector{Int}}()
    for (i, t) in enumerate(triples)
        i_h = head_idx[t[1]]
        i_k = link_idx[t[2]]
        i_t = tail_idx[t[3]]
        triple_ps[:, i] = vcat(head_ps[:, i_h], link_ps[:, i_k], tail_ps[:, i_t])
        if !haskey(h_3_dict, i_h)
            h_3_dict[i_h] = [i]
        else
            push!(h_3_dict[i_h], i)
        end
        if !haskey(t_3_dict, i_t)
            t_3_dict[i_t] = [i]
        else
            push!(t_3_dict[i_t], i)
        end
        ht = "$(t[1]) - * - $(t[3])"
        if !haskey(h_t_3_dict, ht)
            h_t_3_dict[ht] = [i]
        else
            push!(h_t_3_dict[ht], i)
        end
    end
    @info "triple_ps", triple_ps
    hms = Vector{Vector{Int}}(undef, length(h_3_dict))
    for (i, ks) in h_3_dict
        hms[i] = ks
    end
    global h_meta = [pydict(["triple" => join(ks, ' ')]) for (i, ks) in enumerate(hms)]
    @info "head", h_3_dict, h_meta
    tms = Vector{Vector{Int}}(undef, length(t_3_dict))
    for (i, ks) in t_3_dict
        tms[i] = ks
    end
    global t_meta = [pydict(["triple" => join(ks, ' ')]) for (i, ks) in enumerate(tms)]
    @info "tail", t_3_dict, t_meta
    global h_t_ps = Matrix{Float32}(undef, 2 * xdims, length(h_t_3_dict))
    global h_t_docs = Vector{String}()
    htms = Vector{Vector{Int}}()
    for (i, h_t) in enumerate(h_t_3_dict)
        push!(h_t_docs, h_t[1])
        push!(htms, h_t[2])
        t = triples[h_t[2][1]]
        i_h = head_idx[t[1]]
        i_t = tail_idx[t[3]]
        h_t_ps[:, i] = vcat(head_ps[:, i_h], tail_ps[:, i_t])
    end
    global h_t_meta = [pydict(["triple" => join(ks, ' ')]) for (i, ks) in enumerate(htms)]
    @info h_t_3_dict, h_t_ps, h_t_docs, h_t_meta
end

# ╔═╡ 4ae93e2f-c6e0-4ffa-81a9-26569d4644df
begin
    chromadb = pyimport("chromadb")
    client = chromadb.PersistentClient(path="../data/knowledge_graph")
end

# ╔═╡ 991d4933-08d6-4a3e-8553-7179e74fd342
md"""
**Clean up ChromaDB collection:`  `**
$(@bind cleanup_db_colls CheckBox())
"""

# ╔═╡ 65c29682-dcc7-4938-ac4b-d394156a3855
let
    if cleanup_db_colls
        client.delete_collection(name="graph_heads")
        client.delete_collection(name="spell_heads")
        client.delete_collection(name="graph_tails")
        client.delete_collection(name="spell_tails")
        client.delete_collection(name="graph_triples")
        client.delete_collection(name="graph_heads_tails")
    end
end

# ╔═╡ cf3a6651-184c-48c5-a282-e46f8394c624
begin
    head_coll = client.get_or_create_collection("graph_heads")
    h_spell_coll = client.get_or_create_collection("spell_heads")
    tail_coll = client.get_or_create_collection("graph_tails")
    t_spell_coll = client.get_or_create_collection("spell_tails")
    triple_coll = client.get_or_create_collection("graph_triples")
    head_tail_coll = client.get_or_create_collection("graph_heads_tails")
end

# ╔═╡ f0878964-590b-431e-ba42-faa71cc5210e
md"""
**Load graph embedding vectors into db:`  `**
$(@bind load_graph_vs CheckBox())
"""

# ╔═╡ f24d2bef-11dd-4fd1-8436-1b90b13a5806
let
    if load_graph_vs
        head_coll.upsert(
            documents=pylist(heads),
            embeddings=pycollist(head_ps),
            ids=pylist(["$k" for k in 1:n_heads]),
            metadatas=pylist(h_meta)
        )
        h_spell_coll.upsert(
            documents=pylist(heads),
            embeddings=pycollist(h_spell_ps),
            ids=pylist(["$k" for k in 1:n_heads]),
        )
        tail_coll.upsert(
            documents=pylist(tails),
            embeddings=pycollist(tail_ps),
            ids=pylist(["$k" for k in 1:n_tails]),
            metadatas=pylist(t_meta)
        )
        t_spell_coll.upsert(
            documents=pylist(tails),
            embeddings=pycollist(t_spell_ps),
            ids=pylist(["$k" for k in 1:n_tails])
        )
        triple_coll.upsert(
            documents=pylist(["$(t[1]) - $(t[2]) - $(t[3])" for t in triples]),
            embeddings=pycollist(triple_ps),
            ids=pylist(["$k" for k in 1:n_triples])
        )
        head_tail_coll.upsert(
            documents=pylist(h_t_docs),
            embeddings=pycollist(h_t_ps),
            ids=pylist(["$k" for k in 1:length(h_t_docs)]),
            metadatas=pylist(h_t_meta)
        )
    end
end

# ╔═╡ 65021805-4094-43d4-9821-9ce65433859d
function show_results(q_rs)
    rs = pyconvert(Dict, q_rs)
    tds = pyconvert(Vector, rs["documents"][1])
    ds = pyconvert(Vector, rs["distances"][1])
    metas = pyconvert(Vector, rs["metadatas"][1])
    ts = if metas[1] == nothing
        nothing
    else
        s_list = [pyconvert(Dict, t)["triple"] for t in metas]
        split.(s_list)
    end
    for (i, t) in enumerate(tds)
        println(t, "  ", round(ds[i], digits=4))
        if ts != nothing
            ks = parse.(Int, ts[i])
            for k in ks
                p = triples[k]
                println("    $(p[1]) - $(p[2]) - $(p[3])")
            end
        end
    end
end

# ╔═╡ 2a9bf35b-8407-417d-86b3-59d546483016
md"""
**Graph Triple Index:`  `**
$(@bind graph_triple_idx NumberField(1:n_triples, default=1))
"""

# ╔═╡ dabb5d42-187f-4385-b683-4a8b673d8089
let
    t = triples[graph_triple_idx]
    print("$(t[1]) - $(t[2]) - $(t[3])")
end

# ╔═╡ a80afb21-85c3-481e-9948-0a31d0294ed2
let
    top_k = 7
    q_rs = triple_coll.query(
        query_embeddings=pycollist(triple_ps[:, graph_triple_idx]),
        n_results=top_k
    )
    show_results(q_rs)
end

# ╔═╡ f17ac447-8f1e-4ad5-9f21-123ae86e63c1
md"""
**Graph Head Index:`  `**
$(@bind graph_head_idx NumberField(1:n_heads, default=1))
"""

# ╔═╡ 6b783ebc-c324-4988-a9b3-28e4d481e73f
let
    print(heads[graph_head_idx])
end

# ╔═╡ 0ee49c18-a7c6-4059-9352-591f7546f0c4
let
    top_k = 7
    q_rs = head_coll.query(
        query_embeddings=pycollist(head_ps[:, graph_head_idx]),
        n_results=top_k
    )
    show_results(q_rs)
end

# ╔═╡ 3568607c-e695-4752-b096-3443f9fa4817
md"""
**Graph Tail Index:`  `**
$(@bind graph_tail_idx NumberField(1:n_tails, default=1))
"""

# ╔═╡ 940d6782-551a-459e-bc55-424c9eea43bf
let
    print(tails[graph_tail_idx])
end

# ╔═╡ 4d8dba6c-a465-4142-9b63-d468ff336921
let
    top_k = 7
    q_rs = tail_coll.query(
        query_embeddings=pycollist(tail_ps[:, graph_tail_idx]),
        n_results=top_k
    )
    show_results(q_rs)
end

# ╔═╡ 6667b91a-6d7a-45f2-b781-d728f3cd140e
function token_spell_vec(tkn::String, ut::Matrix{Float32}, ch_max::Int=127, idx_lim::Int=5)::Vector{Float32}
    ch_pos = Dict{Int,Vector{Int}}()
    for (i, ch) in enumerate(tkn)
        i_ch = Int(ch)
        if !haskey(ch_pos, i_ch)
            ch_pos[i_ch] = [i]
        else
            push!(ch_pos[i_ch], i)
        end
    end
    N = 32
    m_max = 6
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
    ut * collect(sp_v)
end

# ╔═╡ e560a9c9-4071-48b6-a780-cc1d5b3f8e68
@htl("""
	<div>
		<label>Head input:</label>&nbsp;&nbsp;$(@bind head_input TextField())&nbsp;&nbsp;&nbsp;&nbsp;
		<label>Tail input:</label>&nbsp;&nbsp;$(@bind tail_input TextField())
	</div>
""")

# ╔═╡ bb35dc8b-1bda-486f-a7c9-64518207bb58
let
    x = token_spell_vec(lowercase(head_input), h_spell_ut)
    top_k = 7
    q_rs = h_spell_coll.query(
        query_embeddings=pycollist(x),
        n_results=top_k
    )
    rs = pyconvert(Dict, q_rs)
    ids = pyconvert(Vector, rs["ids"][1])
    ts = pyconvert(Vector, rs["documents"])[1]
    ds = pyconvert(Vector, rs["distances"])[1]
    global h_options = [""]
    global h_option_idx = Dict{String,Int}("" => 0)
    for (i, t) in enumerate(ts)
        d = round(ds[i], digits=4)
        h = "$t  -  $d"
        push!(h_options, h)
        h_option_idx[h] = parse(Int, ids[i])
    end
end

# ╔═╡ acab4461-247e-4835-925e-79130b1bf3aa
let
    x = token_spell_vec(lowercase(tail_input), t_spell_ut)
    top_k = 7
    q_rs = t_spell_coll.query(
        query_embeddings=pycollist(x),
        n_results=top_k
    )
    rs = pyconvert(Dict, q_rs)
    ids = pyconvert(Vector, rs["ids"][1])
    ts = pyconvert(Vector, rs["documents"])[1]
    ds = pyconvert(Vector, rs["distances"])[1]
    global t_options = [""]
    global t_option_idx = Dict{String,Int}("" => 0)
    for (i, t) in enumerate(ts)
        d = round(ds[i], digits=4)
        h = "$t  -  $d"
        push!(t_options, h)
        t_option_idx[h] = parse(Int, ids[i])
    end
end

# ╔═╡ f0365d12-5643-4cfd-9c79-5274153a9b41
begin
    local options = [k => k for k in links]
    push!(options, "" => "")
    @htl("""
    <div>
    	<label>Head options:</label>&nbsp;&nbsp;$(@bind closest_heads MultiSelect(h_options))&nbsp;&nbsp;&nbsp;&nbsp;
    	<label>Tail options:</label>&nbsp;&nbsp;$(@bind closest_tails MultiSelect(t_options))
    </div>
    <br>
    <label>Link:</label>&nbsp;&nbsp;$(@bind link_select Select(options, ""))
    """)
end

# ╔═╡ da4b4397-b30e-41a7-adf1-fa42a40478f7
let
    h_sel = if length(closest_heads) > 0
        h_sel = h_option_idx[closest_heads[1]]
    else
        0
    end
    k_sel = if link_select != ""
        link_idx[link_select]
    else
        0
    end
    t_sel = if length(closest_tails) > 0
        t_sel = t_option_idx[closest_tails[1]]
    else
        0
    end
    top_k = 7
    if h_sel != 0 && k_sel == 0 && t_sel == 0
        println("? $(heads[h_sel]) - * - *")
        q_rs = head_coll.query(
            query_embeddings=pycollist(head_ps[:, h_sel]),
            n_results=top_k
        )
        show_results(q_rs)
    elseif h_sel == 0 && k_sel == 0 && t_sel != 0
        println("? * - * - $(tails[t_sel])")
        q_rs = tail_coll.query(
            query_embeddings=pycollist(tail_ps[:, t_sel]),
            n_results=top_k
        )
        show_results(q_rs)
    elseif h_sel != 0 && k_sel == 0 && t_sel != 0
        println("? $(heads[h_sel]) - * - $(tails[t_sel])")
        h_t = vcat(head_ps[:, h_sel], tail_ps[:, t_sel])
        q_rs = head_tail_coll.query(
            query_embeddings=pycollist(h_t),
            n_results=top_k
        )
        show_results(q_rs)
    elseif h_sel != 0 && k_sel != 0 && t_sel != 0
        println("? $(heads[h_sel]) - $(links[k_sel]) - $(tails[t_sel])")
        t3 = vcat(head_ps[:, h_sel], link_ps[:, k_sel], tail_ps[:, t_sel])
        q_rs = triple_coll.query(
            query_embeddings=pycollist(t3),
            n_results=top_k
        )
        show_results(q_rs)
    else
        println("NA")
    end
end

# ╔═╡ Cell order:
# ╠═76fbeff9-3a2b-48db-bd5e-fb5afce8e701
# ╠═e09d6b44-6370-11ed-1d3e-3fa29ee591a3
# ╠═7d65c9d5-6dd7-45ac-8aee-b0b1d94329b9
# ╠═0ead4006-0a2d-4eb5-bac9-23077729e79b
# ╠═3bb7c7a8-d8cf-4f39-ab31-f7d670ad3b49
# ╠═e9d3aea1-b98f-4da3-aa17-ba2473959fe6
# ╟─607395ef-b0d2-41c5-8159-32d2c91763f2
# ╠═e7549436-2c9d-4edb-88aa-3e443c7be581
# ╠═b8155ffa-8479-424e-83e7-b0c29e938fb6
# ╠═1206c585-fa44-4f1b-9af2-fa12508c82f8
# ╟─438c6e61-1ef2-40f2-b2cf-1ab907ea162d
# ╠═16c0a8af-5580-4269-9b7b-87d87c14325e
# ╠═a8e9e421-7f67-42bd-bfc9-d6dd4fd2ff36
# ╟─d7d6377f-e3d9-43b3-974d-fd43acd07462
# ╠═9a2bb946-c658-49db-a6be-c1a056e4821d
# ╠═52c3206a-a670-401a-9f3f-df4367f64615
# ╠═a3e05bcc-51e3-4c86-8b08-769dae5962fe
# ╠═4ae93e2f-c6e0-4ffa-81a9-26569d4644df
# ╟─991d4933-08d6-4a3e-8553-7179e74fd342
# ╠═65c29682-dcc7-4938-ac4b-d394156a3855
# ╠═cf3a6651-184c-48c5-a282-e46f8394c624
# ╟─f0878964-590b-431e-ba42-faa71cc5210e
# ╠═f24d2bef-11dd-4fd1-8436-1b90b13a5806
# ╠═65021805-4094-43d4-9821-9ce65433859d
# ╟─2a9bf35b-8407-417d-86b3-59d546483016
# ╟─dabb5d42-187f-4385-b683-4a8b673d8089
# ╟─a80afb21-85c3-481e-9948-0a31d0294ed2
# ╟─f17ac447-8f1e-4ad5-9f21-123ae86e63c1
# ╟─6b783ebc-c324-4988-a9b3-28e4d481e73f
# ╟─0ee49c18-a7c6-4059-9352-591f7546f0c4
# ╟─3568607c-e695-4752-b096-3443f9fa4817
# ╟─940d6782-551a-459e-bc55-424c9eea43bf
# ╟─4d8dba6c-a465-4142-9b63-d468ff336921
# ╠═6667b91a-6d7a-45f2-b781-d728f3cd140e
# ╟─e560a9c9-4071-48b6-a780-cc1d5b3f8e68
# ╟─f0365d12-5643-4cfd-9c79-5274153a9b41
# ╟─da4b4397-b30e-41a7-adf1-fa42a40478f7
# ╟─bb35dc8b-1bda-486f-a7c9-64518207bb58
# ╟─acab4461-247e-4835-925e-79130b1bf3aa
