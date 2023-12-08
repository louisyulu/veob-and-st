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
using Images, PlutoUI, Plots, Logging, Dates, LinearAlgebra, RandomizedLinAlg, Statistics, SparseArrays,
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

# ╔═╡ 755f28e4-935d-4bd8-a3ab-95019d043d3e
if !isdir("../data")
    mkdir("../data")
end

# ╔═╡ 98a68fc1-983f-4c8c-a97b-7d487ce370a6
deeplake = pyimport("deeplake")

# ╔═╡ 9d8a119e-b744-4184-8ffb-88a93000a0b8
md"""
**Load digit/fashion MNIST data from Deeplake:`  `**
$(@bind load_deeplake_mnist CheckBox())
"""

# ╔═╡ 3572f500-ce8e-46d8-b94c-260d23c4fb7b
let
    if load_deeplake_mnist
        mnist = deeplake.load("hub://activeloop/mnist-train")
        mnist_t = deeplake.load("hub://activeloop/mnist-test")
        f_mnist = deeplake.load("hub://activeloop/fashion-mnist-train")
        f_mnist_t = deeplake.load("hub://activeloop/fashion-mnist-test")

        digit_train_x = permutedims(pyconvert(Array{Float32}, mnist.tensors["images"].numpy()), [3, 2, 1]) / 255.0f0
        digit_train_y = vec(pyconvert(Array{Int}, mnist.tensors["labels"].numpy()))
        digit_classes = ["$i" for i in 0:9]
        serialize("../data/mnist_digit_train.jls", (digit_train_x, digit_train_y, digit_classes))
        @info digit_train_x, digit_train_y, digit_classes

        digit_test_x = permutedims(pyconvert(Array{Float32}, mnist_t.tensors["images"].numpy()), [3, 2, 1]) / 255.0f0
        digit_test_y = vec(pyconvert(Array{Int}, mnist_t.tensors["labels"].numpy()))
        serialize("../data/mnist_digit_test.jls", (digit_test_x, digit_test_y, digit_classes))
        @info digit_test_x, digit_test_y

        fashion_train_x = permutedims(pyconvert(Array{Float32}, f_mnist.tensors["images"].numpy()), [3, 2, 1]) / 255.0f0
        fashion_train_y = vec(pyconvert(Array{Int}, f_mnist.tensors["labels"].numpy()))
        fashion_classes = ["T-Shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle-boot"]
        @info fashion_train_x, fashion_train_y, fashion_classes
        serialize("../data/mnist_fashion_train.jls", (fashion_train_x, fashion_train_y, fashion_classes))

        fashion_test_x = permutedims(pyconvert(Array{Float32}, f_mnist_t.tensors["images"].numpy()), [3, 2, 1]) / 255.0f0
        fashion_test_y = vec(pyconvert(Array{Int}, f_mnist_t.tensors["labels"].numpy()))
        serialize("../data/mnist_fashion_test.jls", (fashion_test_x, fashion_test_y, fashion_classes))
        @info fashion_test_x, fashion_test_y
    end
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

# ╔═╡ 2440f130-901a-4481-9ec4-3359a7f85bd2
mask2d = cos_2d_mask(28, 28, 28)

# ╔═╡ aeba4a8a-c8b1-421c-b39d-145b4dc13892
function svd_encoding(s_train::Array{Float32}, s_test::Array{Float32})
    n_train = size(s_train, 3)
    cos_train = Array{Float32}(undef, 28, 28, n_train)
    for k in 1:n_train
        for i in 1:28, j in 1:28
            cos_train[j, i, k] = sum(mask2d[:, :, j, i] .* s_train[:, :, k])
        end
    end
    @info cos_train
    u, s, v = svd(reshape(cos_train, 28 * 28, n_train))
    n_test = size(s_test, 3)
    cos_test = Array{Float32}(undef, 28, 28, n_test)
    for k in 1:n_test
        for i in 1:28, j in 1:28
            cos_test[j, i, k] = sum(mask2d[:, :, j, i] .* s_test[:, :, k])
        end
    end
    @info cos_test
    a_test = reshape(cos_test, 28 * 28, n_test)
    s .* v', u' * a_test, s, u'
end

# ╔═╡ 479d1f91-9481-4155-a6d1-88066600ab66
function build_z(x::Vector{Float32}, label::Int, ut::Matrix{Float32})::Vector{Float32}
    xdim = length(x)
    z = zeros(Float32, xdim * 10)
    z[label*xdim+1:(label+1)*xdim] = x
    ut * z
end

# ╔═╡ 9cc17b5b-748d-4b4a-a1cc-0d81abaae5bc
md"""
**Prepare digit svd vectors:`  `**
$(@bind prepare_digit_svd_vecs CheckBox())
"""

# ╔═╡ b8e4e5b1-d163-4eb9-ad04-60492f904fef
let
    if prepare_digit_svd_vecs
        time1 = now()
        p1, p2, s, ut = svd_encoding(digit_train_x, digit_test_x)
        x_dims = 30
        serialize("../data/digit_train_svd.jls", (p1[1:x_dims, :], ut[1:x_dims, :]))
        serialize("../data/digit_test_svd.jls", p2[1:x_dims, :])
        time2 = now()
        @info "digits: $(size(p1)), $(size(p2)), $(size(ut))"
        @info "Used ", time_span(time1, time2, :Minute)
        plot(1:length(s), s, label="Singular Values Accumulation (digit)")
    end
end

# ╔═╡ 32f4606c-969b-4132-b3da-375f5f3d83dc
md"""
**Prepare fashion svd vectors:`  `**
$(@bind prepare_fasion_svd_vecs CheckBox())
"""

# ╔═╡ 4dfe24fb-75b1-4e37-9219-dd6351d83d48
let
    if prepare_fasion_svd_vecs
        time1 = now()
        p1, p2, s, ut = svd_encoding(fashion_train_x, fashion_test_x)
        x_dims = 30
        serialize("../data/fashion_train_svd.jls", (p1[1:x_dims, :], ut[1:x_dims, :]))
        serialize("../data/fashion_test_svd.jls", p2[1:x_dims, :])
        time2 = now()
        @info "fashions: $(size(p1)), $(size(p2)), $(size(ut))"
        @info "Used ", time_span(time1, time2, :Minute)
        plot(1:length(s), s, label="Singular Values (Fashion)")
    end
end

# ╔═╡ 86ba099c-7c0c-4d81-902e-2e5bbf5c1219
function generate_combined(svdprj::Matrix{Float32}, labels::Vector{Int})
    xdims, n = size(svdprj)
    zs = zeros(Float32, xdims * 10, n)
    for i in 1:n
        k = labels[i]
        zs[k*xdims+1:(k+1)*xdims, i] = svdprj[:, i]
    end
    ntop = 100
    u, s, v = rsvd(zs, ntop)
    ut = convert(Matrix{Float32}, u')
    ps = convert(Matrix{Float32}, s .* v')
    ut, ps, s
end

# ╔═╡ 1a8c277e-d958-4f91-a6e1-7a1af037516e
begin
    digit_train_svd, digit_train_svd_ut = deserialize("../data/digit_train_svd.jls")
    digit_test_svd = deserialize("../data/digit_test_svd.jls")
    fashion_train_svd, fashion_train_svd_ut = deserialize("../data/fashion_train_svd.jls")
    fashion_test_svd = deserialize("../data/fashion_test_svd.jls")
    @info size(digit_train_svd), size(digit_test_svd), size(digit_train_svd_ut), size(fashion_train_svd), size(fashion_test_svd), size(fashion_train_svd_ut)
end

# ╔═╡ d0e11884-4933-4974-94b4-222ee9917d98
begin
    const max_level = 9
    const min_samples = 200

    function partition_data(xs::Matrix{Float32}, outfilepath::String)
        nt = create_node_tree(xs)
        populate_nodes!(nt, max_level, min_samples)
        cl = make_cluster!(nt)
        populate_cells!(cl)
        serialize(outfilepath, cl)
    end
end

# ╔═╡ ed5c1d5b-a017-4242-8720-b32d3b61a27c
md"""
**Partiotion data:`  `**
$(@bind partition_sample_data CheckBox())
"""

# ╔═╡ ccee2c87-38d7-4e32-9519-9d2c36f0e1aa
let
    if partition_sample_data
        digit_ut, digit_ps, s = generate_combined(digit_train_svd, digit_train_y)
        serialize("../data/digit_train_ut_ps.jls", (digit_ut, digit_ps))
        @info digit_ut, digit_ps, s
        partition_data(digit_ps, "../data/digit_train_svd_cluster.jls")

        fashion_ut, fashion_ps, s = generate_combined(fashion_train_svd, fashion_train_y)
        serialize("../data/fashion_train_ut_ps.jls", (fashion_ut, fashion_ps))
        @info fashion_ut, fashion_ps, s
        partition_data(fashion_ps, "../data/fashion_train_svd_cluster.jls")

        plot(1:length(s), s, label="Singular Values (fashion_s)")
    end
end

# ╔═╡ e0c2f7ff-ed75-46bb-8b29-8b0785c3d0d7
begin
    digit_ut, digit_ps = deserialize("../data/digit_train_ut_ps.jls")
    digit_cluster = deserialize("../data/digit_train_svd_cluster.jls")
    fashion_ut, fashion_ps = deserialize("../data/fashion_train_ut_ps.jls")
    fashion_cluster = deserialize("../data/fashion_train_svd_cluster.jls")
    @info size(digit_cluster.cells), size(fashion_cluster.cells)
end

# ╔═╡ b29f018a-b03f-4feb-85ef-70398ae42f96
md"""
**Dataset:`  `**
$(@bind dataset Select([:digit_train=>"Digit Train Set", :digit_test=>"Digit Test Set", :fashion_train=>"Fashion Train Set", :fashion_test=>"Fashion Test Set"]))
"""

# ╔═╡ 4fb83635-8126-417d-b235-24418ce94d58
begin
    samples, labels, classes, svd_cluster =
        if dataset == :digit_train
            digit_train_x, digit_train_y, digit_classes, digit_cluster
        elseif dataset == :digit_test
            digit_test_x, digit_test_y, digit_classes, digit_cluster
        elseif dataset == :fashion_train
            fashion_train_x, fashion_train_y, fashion_classes, fashion_cluster
        else
            fashion_test_x, fashion_test_y, fashion_classes, fashion_cluster
        end
    data_len = length(labels)
end

# ╔═╡ cab5de16-aa73-4e2b-ad7d-664fa304d3c5
md"""
**Cell Index:`  `**
$(@bind cell_index NumberField(1:length(svd_cluster.cells), default=1))
"""

# ╔═╡ 1bc64bc8-040e-41d7-8f9c-0cf4e9ee5cd5
let
    samples, labels =
        if dataset == :digit_train || dataset == :digit_test
            digit_train_x, digit_train_y
        else
            fashion_train_x, fashion_train_y
        end
    imgs = []
    lbls = []
    for i in svd_cluster.cells[cell_index].indices
        push!(imgs, Gray.(permutedims(samples[:, :, i], [2, 1])))
        push!(lbls, labels[i])
    end
    @htl("""
    <div style="display: grid; width: 1200px; grid-template-columns: repeat(auto-fill, minmax(80px, 1fr)">
    	$([@htl("<div style='margin: 0px 2px'><label>$(lbls[k])</label><span>$(embed_display(imgs[k]))</span></div>") for k in eachindex(imgs)])
    </div>
    """)
end

# ╔═╡ f0f0b931-c6ea-49bf-a569-f290ebbd1721
md"""
**Show Index:`  `**
$(@bind show_index NumberField(1:data_len, default=1))
"""

# ╔═╡ 0ae65897-d32d-4d8f-9a52-c26af443bca7
md"""
**Show Count:`  `**
$(@bind show_count Slider(1:16, default=3, show_value=true))
"""

# ╔═╡ a3f11269-5eae-4789-b764-89be05d08e46
if isa(show_index, Int) && show_index >= 1 && show_index <= data_len
    lst = []
    for i in show_index:min(show_index + show_count - 1, data_len - show_count + 1)
        push!(lst, (classes[labels[i]+1], Gray.(permutedims(samples[:, :, i], [2, 1])), "index: $i"))
    end
    @htl("""
    <div style="display: grid; width: 1200px; grid-template-columns: repeat(auto-fill, minmax(90px, 1fr)">
    	$([@htl("<div style='margin: 2px'><label>$(t[1])</label><span>$(embed_display(t[2]))</span><label>$(t[3])</label></div>") 
    	for t in lst])
    </div>
    """)
end

# ╔═╡ 8ee3450d-f4fc-4088-8270-76122cdcfab6
md"""
**Recognition Set:**`  `
$(@bind reconstruct_set Select([:digit_train=>"Digit Train Set", :digit_test=>"Digit Test Set", :fashion_train=>"Fashion Train Set", :fashion_test=>"Fashion Test Set"]))
"""

# ╔═╡ 70202e0e-b6bf-4671-acab-b882b0780580
begin
    smpls, svdprj, lbls, sample_classes, cluster, cl_samples, cl_labels, cl_proj, sample_ut = if reconstruct_set == :digit_train
        digit_train_x, digit_train_svd, digit_train_y, digit_classes, digit_cluster, digit_train_x, digit_train_y, digit_train_svd, digit_ut
    elseif reconstruct_set == :digit_test
        digit_test_x, digit_test_svd, digit_test_y, digit_classes, digit_cluster, digit_train_x, digit_train_y, digit_train_svd, digit_ut
    elseif reconstruct_set == :fashion_train
        fashion_train_x, fashion_train_svd, fashion_train_y, fashion_classes, fashion_cluster, fashion_train_x, fashion_train_y, fashion_train_svd, fashion_ut
    elseif reconstruct_set == :fashion_test
        fashion_test_x, fashion_test_svd, fashion_test_y, fashion_classes, fashion_cluster, fashion_train_x, fashion_train_y, fashion_train_svd, fashion_ut
    end
    lbls_len = length(lbls)
    @info "count: ", lbls_len
end

# ╔═╡ ee4c3828-72ec-4c2c-aa6a-8d15e060e190
md"""
**Match Index:`  `**
$(@bind rc_index NumberField(1:lbls_len, default=1))
"""

# ╔═╡ 877f4603-304a-4ccc-a404-ee8653d39894
if isa(rc_index, Int) && rc_index >= 1 && rc_index <= lbls_len
    @htl("""
    	<div style="display: grid">
    		<label>Original: $(lbls[rc_index])</label>
    		<span>$(embed_display(Gray.(permutedims(smpls[:,:,rc_index], [2,1]))))</span>
    		<label>$(lbls[rc_index]) - $(sample_classes[1+lbls[rc_index]])</label>
    	</div>
    """)
end

# ╔═╡ 33eb5919-f876-4da4-adb1-7a7e50d6053c
if isa(rc_index, Int) && rc_index >= 1 && rc_index <= lbls_len
    x = svdprj[:, rc_index]
    indices = zeros(Int, 0)
    for lbl in 0:9
        z = build_z(x, lbl, sample_ut)
        tis, _, _ = top_matches(cluster, z, 1)
        push!(indices, tis[1])
    end
    ntop = 5
    ps = cl_proj[:, indices]
    ds = vec(sum((x .- ps) .^ 2, dims=1))
    ks = sortperm(ds)[1:ntop]
    reconstructed = []
    for i in ks
        k = indices[i]
        push!(reconstructed, (Gray.(permutedims(cl_samples[:, :, k], [2, 1])), ds[i], cl_labels[k]))
    end
    @htl("""
 <label>Matches:</label>
 <div style="display: grid; width: 1200px; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)">
 	$([@htl("<div style='margin: 2px'><span>$(embed_display(r[1]))</span><label>$(r[3]) - $(r[2])</label></div>") 
 	for r in reconstructed])
 </div>
 """)
end

# ╔═╡ 10a17779-8761-4ad3-83a0-8836be1794ef
function verify(dataset::Symbol, ntop::Int)
    svdprj, labels, cluster, cl_labels, cl_proj, sample_ut = if dataset == :digit_train
        digit_train_svd, digit_train_y, digit_cluster, digit_train_y, digit_train_svd, digit_ut
    elseif dataset == :digit_test
        digit_test_svd, digit_test_y, digit_cluster, digit_train_y, digit_train_svd, digit_ut
    elseif dataset == :fashion_train
        fashion_train_svd, fashion_train_y, fashion_cluster, fashion_train_y, fashion_train_svd, fashion_ut
    elseif dataset == :fashion_test
        fashion_test_svd, fashion_test_y, fashion_cluster, fashion_train_y, fashion_train_svd, fashion_ut
    end
    data_len = length(labels)
    mismatches = 0
    file = open("../data/$dataset-missmatch-svd-cluster-$ntop.txt", "w")
    write(file, "Mismatches of $dataset for top $ntop candidates\n")
    write(file, "index, label, candidates\n")
    for i in 1:data_len
        x = svdprj[:, i]
        indices = zeros(Int, 0)
        for lbl in 0:9
            z = build_z(x, lbl, sample_ut)
            tis, _, _ = top_matches(cluster, z, 1)
            push!(indices, tis[1])
        end
        ps = cl_proj[:, indices]
        ds = vec(sum((x .- ps) .^ 2, dims=1))
        ks = sortperm(ds)[1:ntop]
        lbl = labels[i]
        if !in(lbl, cl_labels[indices[ks]])
            mismatches += 1
            @info "miss matches $mismatches index $i label $lbl candidates $(cl_labels[ks])"
            write(file, "$i, $lbl, $(cl_labels[ks])\n")
        end
    end
    rate = round(100 * (data_len - mismatches) / data_len, digits=4)
    msg = "misatches $mismatches among $data_len, correct rate $rate%\n"
    @info msg
    write(file, msg)
    close(file)
end

# ╔═╡ 354e0ef2-83aa-4155-9749-e30fbcb19daf
# verify(:digit_train, 3)

# ╔═╡ 63a68ed1-7e61-4a71-9f2f-4a5b8b5113b3
# verify(:digit_test, 3)

# ╔═╡ e49d5ab4-c78a-4c1b-bcbf-a6b6f330b433
# verify(:fashion_train, 3)

# ╔═╡ 5e2aff79-a704-42a0-8a47-8dbfa5007c98
# verify(:fashion_test, 3)

# ╔═╡ Cell order:
# ╠═76fbeff9-3a2b-48db-bd5e-fb5afce8e701
# ╠═e09d6b44-6370-11ed-1d3e-3fa29ee591a3
# ╠═7d65c9d5-6dd7-45ac-8aee-b0b1d94329b9
# ╠═0ead4006-0a2d-4eb5-bac9-23077729e79b
# ╠═3bb7c7a8-d8cf-4f39-ab31-f7d670ad3b49
# ╠═755f28e4-935d-4bd8-a3ab-95019d043d3e
# ╠═98a68fc1-983f-4c8c-a97b-7d487ce370a6
# ╟─9d8a119e-b744-4184-8ffb-88a93000a0b8
# ╠═3572f500-ce8e-46d8-b94c-260d23c4fb7b
# ╠═49e09074-830c-4253-95f5-bf57c7980fe1
# ╠═2440f130-901a-4481-9ec4-3359a7f85bd2
# ╠═aeba4a8a-c8b1-421c-b39d-145b4dc13892
# ╠═479d1f91-9481-4155-a6d1-88066600ab66
# ╟─9cc17b5b-748d-4b4a-a1cc-0d81abaae5bc
# ╠═b8e4e5b1-d163-4eb9-ad04-60492f904fef
# ╟─32f4606c-969b-4132-b3da-375f5f3d83dc
# ╠═4dfe24fb-75b1-4e37-9219-dd6351d83d48
# ╠═86ba099c-7c0c-4d81-902e-2e5bbf5c1219
# ╠═1a8c277e-d958-4f91-a6e1-7a1af037516e
# ╠═d0e11884-4933-4974-94b4-222ee9917d98
# ╟─ed5c1d5b-a017-4242-8720-b32d3b61a27c
# ╠═ccee2c87-38d7-4e32-9519-9d2c36f0e1aa
# ╠═e0c2f7ff-ed75-46bb-8b29-8b0785c3d0d7
# ╟─b29f018a-b03f-4feb-85ef-70398ae42f96
# ╟─4fb83635-8126-417d-b235-24418ce94d58
# ╟─cab5de16-aa73-4e2b-ad7d-664fa304d3c5
# ╟─1bc64bc8-040e-41d7-8f9c-0cf4e9ee5cd5
# ╟─f0f0b931-c6ea-49bf-a569-f290ebbd1721
# ╟─0ae65897-d32d-4d8f-9a52-c26af443bca7
# ╟─a3f11269-5eae-4789-b764-89be05d08e46
# ╟─8ee3450d-f4fc-4088-8270-76122cdcfab6
# ╟─70202e0e-b6bf-4671-acab-b882b0780580
# ╟─ee4c3828-72ec-4c2c-aa6a-8d15e060e190
# ╟─877f4603-304a-4ccc-a404-ee8653d39894
# ╟─33eb5919-f876-4da4-adb1-7a7e50d6053c
# ╠═10a17779-8761-4ad3-83a0-8836be1794ef
# ╠═354e0ef2-83aa-4155-9749-e30fbcb19daf
# ╠═63a68ed1-7e61-4a71-9f2f-4a5b8b5113b3
# ╠═e49d5ab4-c78a-4c1b-bcbf-a6b6f330b433
# ╠═5e2aff79-a704-42a0-8a47-8dbfa5007c98
