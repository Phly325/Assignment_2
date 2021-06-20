### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 72fbc049-d390-49ad-862f-80ae510da7e5
using Pkg;Pkg.add("NORMAL");Pkg.add("PNEUMONIA");Pkg.add("Dataset");Pkg.add("Plots");Pkg.add("Flux")

# ╔═╡ 988b4606-c626-4c22-bc27-2d3b5d6a535e
Pkg.add("GPUArrays")

# ╔═╡ a1e3b8f3-5368-415a-bc17-7e03802a010f
using Flux

# ╔═╡ 90166fe6-351c-4638-befc-bfe1b353181d
using Flux: onehotbatch, onecold, crossentropy, throttle

# ╔═╡ c80b3806-eea6-471e-aa09-874943fb3cce
using partition

# ╔═╡ b30423d7-5bf4-4c68-bc0e-96f3ea4bddea
using Printf

# ╔═╡ ba84e906-9031-4599-89df-8cbd3b62cc1d
training_labels = Dataset.labels()
training_images = load("C:\Users\ugulu\OneDrive\Documents\3rd YEAH!\AIG710S\Assignment_2\Dataset\training_set")

# ╔═╡ 1ea22d71-c44b-407e-8ca0-95ae5abe8df4
training_images = load("C:\Users\ugulu\OneDrive\Documents\3rd YEAH!\AIG710S\Assignment_2\Dataset\training_set")(:test)
training_labels = Dataset.labels()(:test)
test_set = smallbatch(testing_images, test_labels, 1:length(testing_images))

# ╔═╡ 5525637a-cc77-4523-9d15-b72053211d62
train_set[1][2][:,1]

# ╔═╡ 8496b2f7-4b27-42dc-8d49-89e8a8f57116
testing_set = gpu.(testing_set)

# ╔═╡ 4e75b165-8ee4-4ffa-943d-a6d3f3981953
typeof(training_set)

# ╔═╡ ba59cce3-e415-4c1c-90f4-a5f13c3a5269
size(training_set[1][1])

# ╔═╡ 853e5c45-f0a5-4b90-9cac-44699bb28706
size(training_set[1][2])

# ╔═╡ 377a27d9-3a48-4f9b-b693-1720e99059ec
model(training_set[1][1])

# ╔═╡ 5f65fe15-a5f8-4968-af3d-6577cb06a3dc
function loss(x, y)
    x_era = x .+ 0.1f0*gpu(randn(eltype(x), size(x)))

    y_omb = model(x_era)
    return crossentropy(y_omb, y)
end

# ╔═╡ fc870935-58d7-4045-9995-b099486545ee
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

# ╔═╡ 4a7c0567-d601-4dd8-b82d-f47fb24c1481
opt = ADAM(0.001)

# ╔═╡ c681868f-aedc-4ddc-961d-b20656f1ea00
test_image = image.load_img("C:\Users\ugulu\OneDrive\Documents\3rd YEAH!\AIG710S\Assignment_2\Dataset\training_set\PNEUMONIA\PNEUMONIA_902.png", target_size = (232, 232))
test_image = image.img_to_array(test_image)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = "NORMAL"
else:
    prediction = "PNUEMONIA"

# ╔═╡ 1ce582d6-3911-4cde-83ee-ba0a02c05c7b
for epoch_idx in 1:100
    global last_chance
    
    Flux.train!(loss, params(model), training_set, opt)

    acc = accuracy(testing_set...)
    @info(@printf("[%d]: Test accuracy: %.4f", epoch_idx, acc))
    
    if acc >= 0.999
        @info(" -> Accuracy of 99.9% reached")
        break
    end
	
    if epoch_idx - last_chance >= 5 && opt.eta > 1e-6
        opt.eta /= 10.0
        @warn(" -> Haven't improved in a while, drop the learning rate to $(opt.eta)!")

        # After dropping learning rate, give it a few epochs to improve
        last_chance = epoch_idx
    end

    if epoch_idx - last_chance >= 10
        @warn(" -> This is Converged.")
        break
    end
end

# ╔═╡ 1c58c7ea-bf7c-48bd-aefb-3ba18fc5d388
training_set = gpu.(training_set)

# ╔═╡ b2a81949-a16e-40bc-bb22-547b308dea8c
begin
	model = Chain(
	    Conv((3, 3), 1=>16, pad=(1,1), relu),
	    x -> maxpool(x, (2,2)),
		
	    Conv((3, 3), 16=>32, pad=(1,1), relu),
	    x -> maxpool(x, (2,2)),
	
	    Conv((3, 3), 32=>32, pad=(1,1), relu),
	    x -> maxpool(x, (2,2)),
		
	    x -> reshape(x, :, size(x, 4)),
	    Dense(288, 10),
	
	    softmax,
	)
end

# ╔═╡ 2d358470-5c7b-4f65-a975-b11b127d7a75
last_chance = 0

# ╔═╡ 401cbe2d-5a6b-4234-8b64-26572c690c3d
model = gpu(model)

# ╔═╡ c99fa82d-8ab4-48e0-960b-feafb34bf8dd
begin
	function smallbatch(X, Y, idxs)
	    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
	    for i in 1:length(idxs)
	        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
	    end
	    Y_batch = onehotbatch(Y[idxs], 0:9)
	    return (X_batch, Y_batch)
	end
	batch_size = 128
	sb_idxs = partition(1:length(training_images), batch_size)
	training_set = [smallbatch(training_images, training_labels, i) for i in sb_idxs]
	
end

# ╔═╡ 009bdb80-6eba-4d4b-a99d-80a5a98efb9a
using training_set; testing_set; NORMAL; PNEUMONIA; Dataset; Plots; gpu

# ╔═╡ Cell order:
# ╠═72fbc049-d390-49ad-862f-80ae510da7e5
# ╠═988b4606-c626-4c22-bc27-2d3b5d6a535e
# ╠═009bdb80-6eba-4d4b-a99d-80a5a98efb9a
# ╠═a1e3b8f3-5368-415a-bc17-7e03802a010f
# ╠═90166fe6-351c-4638-befc-bfe1b353181d
# ╠═c80b3806-eea6-471e-aa09-874943fb3cce
# ╠═b30423d7-5bf4-4c68-bc0e-96f3ea4bddea
# ╠═ba84e906-9031-4599-89df-8cbd3b62cc1d
# ╠═c99fa82d-8ab4-48e0-960b-feafb34bf8dd
# ╠═1ea22d71-c44b-407e-8ca0-95ae5abe8df4
# ╠═4e75b165-8ee4-4ffa-943d-a6d3f3981953
# ╠═ba59cce3-e415-4c1c-90f4-a5f13c3a5269
# ╠═853e5c45-f0a5-4b90-9cac-44699bb28706
# ╠═5525637a-cc77-4523-9d15-b72053211d62
# ╠═b2a81949-a16e-40bc-bb22-547b308dea8c
# ╠═1c58c7ea-bf7c-48bd-aefb-3ba18fc5d388
# ╠═8496b2f7-4b27-42dc-8d49-89e8a8f57116
# ╠═401cbe2d-5a6b-4234-8b64-26572c690c3d
# ╠═377a27d9-3a48-4f9b-b693-1720e99059ec
# ╠═5f65fe15-a5f8-4968-af3d-6577cb06a3dc
# ╠═fc870935-58d7-4045-9995-b099486545ee
# ╠═4a7c0567-d601-4dd8-b82d-f47fb24c1481
# ╠═2d358470-5c7b-4f65-a975-b11b127d7a75
# ╠═1ce582d6-3911-4cde-83ee-ba0a02c05c7b
# ╠═c681868f-aedc-4ddc-961d-b20656f1ea00