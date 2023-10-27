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
using Images, PlutoUI, Plots, Logging, Dates, Serialization, HypertextLiteral, PythonCall, Printf

# ╔═╡ e6561128-4b3f-4e7c-b433-29a046ceb47c
begin
	include("../src/my_utils.jl")
    chromadb = pyimport("chromadb")
    client = chromadb.PersistentClient(path="../data/mnistdb")
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

# ╔═╡ afe6bf96-601d-4112-8ce7-36c637a9f217
if !isdir("../data/mnistdb")
	mkdir("../data/mnistdb")
end

# ╔═╡ bf777d3f-1a99-412f-b50d-a0e223d25381
begin
    digit_coll = client.get_or_create_collection("digit_train_coll")
    fashion_coll = client.get_or_create_collection("fashion_train_coll")
end

# ╔═╡ 94bccf6f-adfc-4146-a8c8-e58078d9cdf1
let
    # client.delete_collection(name="digit_train_coll")
    # client.delete_collection(name="fashion_train_coll")
end

# ╔═╡ 49e09074-830c-4253-95f5-bf57c7980fe1
begin
    digit_train_x, digit_train_y, digit_classes = deserialize("../data/mnist_digit_train.jls")
    @info digit_train_x, digit_train_y, digit_classes

    digit_test_x, digit_test_y = deserialize("../data/mnist_digit_test.jls")
    @info digit_test_x, digit_test_y

    fashion_train_x, fashion_train_y, fashion_classes = deserialize("../data/mnist_fashion_train.jls")
    @info fashion_train_x, fashion_train_y, fashion_classes

    fashion_test_x, fashion_test_y = deserialize("../data/mnist_fashion_test.jls")
    @info fashion_test_x, fashion_test_y
end

# ╔═╡ 1a8c277e-d958-4f91-a6e1-7a1af037516e
begin
    digit_train_svd, digit_train_svd_ut = deserialize("../data/digit_train_svd.jls")
    n_vs = size(digit_train_svd, 2)
    digit_test_svd = deserialize("../data/digit_test_svd.jls")
    fashion_train_svd, fashion_train_svd_ut = deserialize("../data/fashion_train_svd.jls")
    fashion_test_svd = deserialize("../data/fashion_test_svd.jls")
    @info size(digit_train_svd), size(digit_test_svd), size(digit_train_svd_ut), size(fashion_train_svd), size(fashion_test_svd), size(fashion_train_svd_ut)
end

# ╔═╡ 7c240bf1-c651-495f-a30a-2750dd4ddd69
md"""
**Save digit/fashing embedding into db:`  `**
$(@bind save_embedding CheckBox())
\
Check the box to run the following cell once, then unckeck it after the run
"""

# ╔═╡ 67775396-48fb-46bf-ae89-76575568d210
let
    if save_embedding
        time1 = now()
        i = 1
        n_batch = 1000
        while i < n_vs
            i2 = if i + n_batch - 1 < n_vs
                i + n_batch - 1
            else
                n_vs
            end
            @info i, i2
            docs = [digit_classes[1+digit_train_y[j]] for j in i:i2]
            digit_coll.upsert(
                documents=pylist(docs),
                embeddings=pycollist(digit_train_svd[:, i:i2]),
                ids=pylist(["$k" for k in i:i2])
            )
            docs = [fashion_classes[1+fashion_train_y[j]] for j in i:i2]
            fashion_coll.upsert(
                documents=pylist(docs),
                embeddings=pycollist(fashion_train_svd[:, i:i2]),
                ids=pylist(["$k" for k in i:i2])
            )
            i = i2 + 1
        end
        time2 = now()
        @info "Used ", time_span(time1, time2, :Minute)
        @info digit_coll.peek(1), fashion_coll.peek(1)
    end
end

# ╔═╡ 8ee3450d-f4fc-4088-8270-76122cdcfab6
md"""
**Query Set:**`  `
$(@bind query_set Select([:digit_train=>"Digit Train Set", :digit_test=>"Digit Test Set", :fashion_train=>"Fashion Train Set", :fashion_test=>"Fashion Test Set"]))
"""

# ╔═╡ ee4c3828-72ec-4c2c-aa6a-8d15e060e190
begin
	lbls_len = if query_set == :digit_train
		length(digit_train_y)
	elseif query_set == :digit_test
		length(digit_test_y)
	elseif query_set == :fashion_train
		length(fashion_train_y)
	else
		length(fashion_test_y)
	end
	md"""
	**Quey Index:`  `**
	$(@bind query_idx NumberField(1:lbls_len, default=1))
	"""
end

# ╔═╡ 39c0d6c8-37f0-49f4-84f9-961b7142177a
let
	selected, title = if query_set == :digit_train
		digit_train_x[:, :, query_idx], digit_classes[1+digit_train_y[query_idx]]
	elseif query_set == :digit_test
		digit_test_x[:, :, query_idx], digit_classes[1+digit_test_y[query_idx]]
	elseif query_set == :fashion_train
		fashion_train_x[:, :, query_idx], fashion_classes[1+fashion_train_y[query_idx]]
	else
	    fashion_test_x[:, :, query_idx], fashion_classes[1+fashion_test_y[query_idx]]
	end
	img = Gray.(permutedims(selected, [2,1]))
	@htl("""
	 <label>$(title)</label>
	 <div style="display: grid; width: 1200px; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)">
	 	$(@htl("<div style='margin: 2px'><span>$(embed_display(img))</span></div>"))
	 </div>""")
end

# ╔═╡ 33eb5919-f876-4da4-adb1-7a7e50d6053c
if isa(query_idx, Int) && query_idx >= 1 && query_idx <= lbls_len
	x, coll, samples = if query_set == :digit_train
		digit_train_svd[:, query_idx], digit_coll, digit_train_x
	elseif query_set == :digit_test
		digit_test_svd[:, query_idx], digit_coll, digit_train_x
	elseif query_set == :fashion_train
		fashion_train_svd[:, query_idx], fashion_coll, fashion_train_x
	else
	    fashion_test_svd[:, query_idx], fashion_coll, fashion_train_x
	end
	
    top_k = 7
    q_rs = coll.query(
        query_embeddings=pycollist(x),
        n_results=top_k
    )
    rs = pyconvert(Dict, q_rs)
	ids = pyconvert(Vector, rs["ids"][1])
    lbls = pyconvert(Vector, rs["documents"][1])
    ds = pyconvert(Vector, rs["distances"][1])
    matches = []
    for (i, s) in enumerate(ids)
        k = parse(Int, s)
        push!(matches, (Gray.(permutedims(samples[:, :, k], [2, 1])), lbls[i], @sprintf("%.6f", ds[i])))
    end
    @htl("""
	 <label>Closest matches:</label>
	 <div style="display: grid; width: 1200px; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)">
	 	$([@htl("<div style='margin: 2px'><span>$(embed_display(r[1]))</span><span>$(r[2])<br>$(r[3])</span></div>") 
	 	for r in matches])
	 </div>
	 """)
end

# ╔═╡ ce57797b-c387-4460-9e59-f5fe6fb541b3
md"""
**Check the dataset for mismatch:`  `**
$(@bind check_mismatch CheckBox())
\
Check the box to run the following cell once, then unckeck it after the run
"""

# ╔═╡ bc81988f-88b9-4e8b-acd7-7184c64d33d0
let
	if check_mismatch
		### check these 2 parameters to run different tests
		top_k = 3
		testing_set = :digit_test
		###
		svd, coll, labels, cls = if testing_set == :digit_train
			digit_train_svd, digit_coll, digit_train_y, digit_classes
		elseif testing_set == :digit_test
			digit_test_svd, digit_coll, digit_test_y, digit_classes
		elseif testing_set == :fashion_train
			fashion_train_svd, fashion_coll, fashion_train_y, fashion_classes
		else
		    fashion_test_svd, fashion_coll, fashion_test_y, fashion_classes
		end
		n_tests = length(labels)
		@info top_k, testing_set, n_tests
		file = open("../data/$testing_set-mismatch-$top_k.txt", "w")
	    write(file, "Mismatches of $testing_set for top $top_k candidates\n")
	    write(file, "index, label, candidates\n")
		mismatches = 0
		for i in 1:n_tests
			if i % 100 == 1 || i == n_tests
				@info i
			end
			q_rs = coll.query(
		        query_embeddings=pycollist(svd[:, i]),
		        n_results=top_k
		    )
		    rs = pyconvert(Dict, q_rs)
		    lbls = pyconvert(Vector, rs["documents"][1])
			lbl = cls[1+labels[i]]
			if !in(lbl, lbls)
				@info i, lbl, lbls
				write(file, "$i, $lbl, $lbls\n")
				mismatches += 1
			end
		end
		rate = round(100 * (n_tests - mismatches) / n_tests, digits=4)
	    msg = "mismatches $mismatches among $n_tests, correct rate $rate%\n"
	    @info msg
	    write(file, msg)
	    close(file)
	end
end

# ╔═╡ Cell order:
# ╠═76fbeff9-3a2b-48db-bd5e-fb5afce8e701
# ╠═e09d6b44-6370-11ed-1d3e-3fa29ee591a3
# ╠═7d65c9d5-6dd7-45ac-8aee-b0b1d94329b9
# ╠═3bb7c7a8-d8cf-4f39-ab31-f7d670ad3b49
# ╠═afe6bf96-601d-4112-8ce7-36c637a9f217
# ╠═e6561128-4b3f-4e7c-b433-29a046ceb47c
# ╠═bf777d3f-1a99-412f-b50d-a0e223d25381
# ╠═94bccf6f-adfc-4146-a8c8-e58078d9cdf1
# ╠═49e09074-830c-4253-95f5-bf57c7980fe1
# ╠═1a8c277e-d958-4f91-a6e1-7a1af037516e
# ╟─7c240bf1-c651-495f-a30a-2750dd4ddd69
# ╠═67775396-48fb-46bf-ae89-76575568d210
# ╟─8ee3450d-f4fc-4088-8270-76122cdcfab6
# ╟─ee4c3828-72ec-4c2c-aa6a-8d15e060e190
# ╟─39c0d6c8-37f0-49f4-84f9-961b7142177a
# ╟─33eb5919-f876-4da4-adb1-7a7e50d6053c
# ╟─ce57797b-c387-4460-9e59-f5fe6fb541b3
# ╠═bc81988f-88b9-4e8b-acd7-7184c64d33d0
