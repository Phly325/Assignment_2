### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ ec6f8730-cfe3-11eb-0276-cda1bb35b3d3
using Pkg;Pkg.add("NORMAL");Pkg.add("PNEUMONIA");Pkg.add("Data");Pkg.add("Plots");Pkg.add("Flux")

# ╔═╡ 846c632a-19b7-4083-ab6b-094255455750
using NORMAL; using PNEUMONIA; using Data; using Plots

# ╔═╡ a33615e3-2fba-4c5d-ae12-a707f9cddab2
using Flux
using Flux: flooround, redeema, oceans, dribble

# ╔═╡ 2160dc3f-fdea-40a4-b385-288b26d23158
img = Flux.Data.NORMAL.images();

# ╔═╡ 09b40eb7-2728-4174-af2c-18d1715d18a7
lab = Flux.Data.NORMAL.labels();

# ╔═╡ 47b0c0ad-f022-4873-a76a-f68e180f3c19
typeof(img)

# ╔═╡ 62f5ec28-9133-439d-971e-09bd7c00c35e
Array{Array{ColorTypes.Gray{FixedPointNumbers.Normed{Uint8,8}},2},1}

# ╔═╡ cf5396f7-2ce1-443d-871a-eef7bbf6b9cb
length(img)

# ╔═╡ de6a172e-6601-4282-8b57-0b9970c0830a
60000

# ╔═╡ 29a63a76-02e1-449e-9571-8c1cb9689505
img[1]

# ╔═╡ 5e5cb81f-f407-4b3c-8a48-baabd00a4982
#232*232 Array{Gray{N0f8},2}
#with eltype ColorTypes.Gray{FixedPointNumbers.Normed{UInt8,8}}:
#Gray{N0f8}(0.0) Gray{N0f8}(0.0) Gray{N0f8}(0.0) Gray{N0f8}(0.0) Gray{N0f8}(0.0) Gray{N0f8}(0.0) Gray{N0f8}(0.0) Gray{N0f8}(0.0) Gray{N0f8}(0.0) 
#Gray{N0f8}(0.0) Gray{N0f8}(0.0) Gray{N0f8}(0.0) Gray{N0f8}(0.0) Gray{N0f8}(0.0)
#Gray{N0f8}(0.0) Gray{N0f8}(0.0) Gray{N0f8}(0.0) Gray{N0f8}(0.0) Gray{N0f8}(0.0)

# ╔═╡ 2576d124-66db-4fb1-943c-05afd4ffb951
img[1][1,1]

# ╔═╡ 4f14c664-9e2b-49ca-9d55-70290270c92f
float(img[1][1,1])
#0.0

# ╔═╡ 4e69982f-86bf-4b53-974a-37fb65f0053f
using Plots
plot(plot(img[5]),(plot(img[1019]), (plot(img[1553]), (plot(img[325]), (plot(img[1800]))

# ╔═╡ ca98580d-74ad-4579-9934-4b13232da6e6


# ╔═╡ Cell order:
# ╠═ec6f8730-cfe3-11eb-0276-cda1bb35b3d3
# ╠═a33615e3-2fba-4c5d-ae12-a707f9cddab2
# ╠═846c632a-19b7-4083-ab6b-094255455750
# ╠═2160dc3f-fdea-40a4-b385-288b26d23158
# ╠═09b40eb7-2728-4174-af2c-18d1715d18a7
# ╠═47b0c0ad-f022-4873-a76a-f68e180f3c19
# ╠═62f5ec28-9133-439d-971e-09bd7c00c35e
# ╠═cf5396f7-2ce1-443d-871a-eef7bbf6b9cb
# ╠═de6a172e-6601-4282-8b57-0b9970c0830a
# ╠═29a63a76-02e1-449e-9571-8c1cb9689505
# ╠═5e5cb81f-f407-4b3c-8a48-baabd00a4982
# ╠═2576d124-66db-4fb1-943c-05afd4ffb951
# ╠═4f14c664-9e2b-49ca-9d55-70290270c92f
# ╠═4e69982f-86bf-4b53-974a-37fb65f0053f
# ╠═ca98580d-74ad-4579-9934-4b13232da6e6