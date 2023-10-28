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

# ╔═╡ e6654fb7-f186-44e8-b65b-64edcd714ce3
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
##### Load IMDB movie review dataset and extract letters only words and sentences
"""

# ╔═╡ 67dbbc91-8061-4f44-aeeb-8734f8153f22
begin
    spacy = pyimport("spacy")
    nlp = spacy.load("en_core_web_sm")
end

# ╔═╡ 7020bb31-2f74-4346-85b8-16fb946fb946
function add_sents_words(doc, sent_list::Vector{Vector{String}}, word_set::Set{String},
    src::String, sent_src::Vector{String}
)
    for sent in doc.sents
        wds = String[]
        for token in sent
            if pyconvert(Bool, token.is_ascii)
                w = pyconvert(String, token.norm_)
                push!(wds, w)
            end
        end
        # pick sentences at maximum 32 tokens
        if 0 < length(wds) <= 32
            push!(sent_list, wds)
            union!(word_set, wds)
            push!(sent_src, src)
        end
    end
end

# ╔═╡ 1f46c8b2-b691-4a43-b2f8-5eaa3ee45048
md"""
**Extract words and sentences from the original IMDB files:`  `**
$(@bind extract_word_sent CheckBox())
"""

# ╔═╡ 0d7b7d43-e337-45a8-a77e-18a8c868db39
let
    if extract_word_sent
        time1 = now()
        f_dir = "../text/aclImdb/train/unsup"
        f_names = readdir(f_dir)
        sents = Vector{String}[]
        word_set = Set{String}()
        sent_src = String[]
        for (i, f_n) in enumerate(f_names)
            if i % 500 == 1
                @info i, f_n
            end
            fd = open("$f_dir/$f_n", "r")
            f_txt = replace(read(fd, String), !isascii => '~')
            txt = replace(f_txt, r"<[^>]*>" => "", "." => ". ", "," => ", ", "!" => "! ", "?" => "? ", "-" => " - ", "(" => " (", ")" => " )", "\\" => "")
            close(fd)
            doc = nlp(txt)
            add_sents_words(doc, sents, word_set, f_n, sent_src)
        end
        words = collect(word_set)
        word_set = nothing
        word_idx = Dict{String,Int}()
        for (i, w) in enumerate(words)
            word_idx[w] = i
        end
        time2 = now()
        @info "Used ", time_span(time1, time2, :Minute)
        serialize("../data/imdb_words_sents.jls", (words, word_idx, sents, sent_src))
        @info "Words: ", length(words), "Sentences:", length(sents), "Files", length(f_names)
        @info words, word_idx, sents, sent_src
    end
end

# ╔═╡ ab7897fa-e120-4b2f-8148-cc1249dad358
let
    global words, word_idx, sents, sent_src = deserialize("../data/imdb_words_sents.jls")
    global n_words = length(words)
    global n_sents = length(sents)
    @info n_words, words, word_idx, n_sents, sents, sent_src
    max_wds = 0
    min_wds = 10_000
    i_max = 0
    i_min = 0
    n_total = 0
    for (i, sent) in enumerate(sents)
        nw = length(sent)
        n_total += nw
        if nw > max_wds
            max_wds = nw
            i_max = i
        end
        if nw < min_wds
            min_wds = nw
            i_min = i
        end
    end
    @info "Sentence statistics:", n_sents, n_total / n_sents
    s_min = join(sents[i_min], " ")
    @info "Sentence min:", min_wds, i_min, s_min
    s_max = join(sents[i_max], " ")
    @info "Sentence max:", max_wds, i_max, s_max
    l_words = [length(w) for w in words]
    @info "Word statistics:", n_words, mean(l_words), std(l_words)
end

# ╔═╡ afe26287-f3e4-4100-981b-f272a5c04ade
let
    # test string conversion
    txt = """this is a test contains the html tags <br>...<br />. some other symbols(letters and numbers).special\\" chars.
    termination of sentence with!and questions?no space between comma,in -the- expression
    """
    @info txt
    doc = nlp(txt)
    @info pyconvert(Vector, pylist(doc.sents))
    txt2 = replace(txt, r"<[^>]*>" => "", "." => ". ", "," => ", ", "!" => "! ", "?" => "? ", "-" => " - ", "(" => " (", ")" => " )", "\\" => "")
    @info txt2
    doc2 = nlp(txt2)
    @info pyconvert(Vector, pylist(doc2.sents))
end

# ╔═╡ Cell order:
# ╠═76fbeff9-3a2b-48db-bd5e-fb5afce8e701
# ╠═e09d6b44-6370-11ed-1d3e-3fa29ee591a3
# ╠═7d65c9d5-6dd7-45ac-8aee-b0b1d94329b9
# ╠═0ead4006-0a2d-4eb5-bac9-23077729e79b
# ╠═3bb7c7a8-d8cf-4f39-ab31-f7d670ad3b49
# ╠═e6654fb7-f186-44e8-b65b-64edcd714ce3
# ╟─4f321908-9978-46bb-a95e-f12331d652ce
# ╠═67dbbc91-8061-4f44-aeeb-8734f8153f22
# ╠═7020bb31-2f74-4346-85b8-16fb946fb946
# ╟─1f46c8b2-b691-4a43-b2f8-5eaa3ee45048
# ╠═0d7b7d43-e337-45a8-a77e-18a8c868db39
# ╠═ab7897fa-e120-4b2f-8148-cc1249dad358
# ╠═afe26287-f3e4-4100-981b-f272a5c04ade
